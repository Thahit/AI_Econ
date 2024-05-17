import argparse
import logging
import os
import sys
import yaml

from utils import remote, saving

logging.basicConfig(filename='log.txt', format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir", type=str, help="Path to the directory for this run."
    )

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_directory, run_configuration

def set_up_dirs_and_maybe_restore(run_directory, run_configuration, trainer_obj):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_a_ok = saving.load_snapshot(
            trainer_obj, run_directory, load_latest=True
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_a_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        training_step_last_ckpt = (
            int(trainer_obj._timesteps_total) if trainer_obj._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainer_obj._episodes_total) if trainer_obj._episodes_total else 0
        )

    else:
        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # For new runs, load only tf checkpoint weights
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_tf_weights_agents", ""
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents TF weights...")
            saving.load_tf_model_weights(trainer_obj, starting_weights_path_agents)
        else:
            logger.info("Starting with fresh agent TF weights.")

        starting_weights_path_planner = run_configuration["general"].get(
            "restore_tf_weights_planner", ""
        )
        if starting_weights_path_planner:
            logger.info("Restoring planner TF weights...")
            saving.load_tf_model_weights(trainer_obj, starting_weights_path_planner)
        else:
            logger.info("Starting with fresh planner TF weights.")

    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )


def maybe_sync_saez_buffer(trainer_obj, result_dict, run_configuration):
    if result_dict["episodes_this_iter"] == 0:
        return

    # This logic just detects if we're using the Saez formula
    sync_saez = False
    for component in run_configuration["env"]["components"]:
        assert isinstance(component, dict)
        c_name = list(component.keys())[0]
        c_kwargs = list(component.values())[0]
        if c_name in ["PeriodicBracketTax"]:
            tax_model = c_kwargs.get("tax_model", "")
            if tax_model == "saez":
                sync_saez = True
                break

    # Do the actual syncing
    if sync_saez:
        remote.accumulate_and_broadcast_saez_buffers(trainer_obj)


def maybe_store_dense_log(
    trainer_obj, result_dict, dense_log_freq, dense_log_directory
):
    if result_dict["episodes_this_iter"] > 0 and dense_log_freq > 0:
        episodes_per_replica = (
            result_dict["episodes_total"] // result_dict["episodes_this_iter"]
        )
        if episodes_per_replica == 1 or (episodes_per_replica % dense_log_freq) == 0:
            log_dir = os.path.join(
                dense_log_directory,
                "logs_{:016d}".format(result_dict["timesteps_total"]),
            )
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            saving.write_dense_logs(trainer_obj, log_dir)
            logger.info(">> Wrote dense logs to: %s", log_dir)


def maybe_save(trainer_obj, result_dict, ckpt_freq, ckpt_directory, trainer_step_last_ckpt):
    global_step = result_dict["timesteps_total"]

    # Check if saving this iteration
    if (
        result_dict["episodes_this_iter"] > 0
    ):  # Don't save if midway through an episode.

        if ckpt_freq > 0:
            if global_step - trainer_step_last_ckpt >= ckpt_freq:
                saving.save_snapshot(trainer_obj, ckpt_directory, suffix="")
                saving.save_tf_model_weights(
                    trainer_obj, ckpt_directory, global_step, suffix="agent"
                )
                saving.save_tf_model_weights(
                    trainer_obj, ckpt_directory, global_step, suffix="planner"
                )

                trainer_step_last_ckpt = int(global_step)

                logger.info("Checkpoint saved @ step %d", global_step)

    return trainer_step_last_ckpt
