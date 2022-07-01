import os
import sys
import subprocess

try: import neptune.new as neptune
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neptune-client"])
    import neptune.new as neptune


def plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args):
    run = neptune.init(project=NEPTUNE_PROJECT,
                       api_token=NEPTUNE_API_KEY,
                       mode="offline",
                       )

    run["sys/name"] = "yolo-optimization"
    run["sys/tags"].add([f"{key}: {value}" for key, value in args._get_kwargs()])

    return run


def record_train_loss(run, loss, total_loss):
    run["train/loss/yx_loss"].log(loss[0].numpy())
    run["train/loss/hw_loss"].log(loss[1].numpy())
    run["train/loss/obj_loss"].log(loss[2].numpy())
    run["train/loss/nobj_loss"].log(loss[3].numpy())
    run["train/loss/cls_loss"].log(loss[4].numpy())
    run["train/loss/total_loss"].log(total_loss.numpy())


def sync_neptune(run, experiment_name, mean_ap, train_time, mean_test_time, NEPTUNE_API_KEY, NEPTUNE_PROJECT):
    res = {
        "mean_ap": "%.3f" % (mean_ap.numpy()),
        "train_time": train_time,
        "inference_time": "%.2fms" % (mean_test_time.numpy()),
    }
    run["results"] = res

    os.environ["NEPTUNE_API_TOKEN"] = NEPTUNE_API_KEY
    cmd = f"neptune sync -p {NEPTUNE_PROJECT} --run offline/run__{experiment_name}"
    subprocess.check_output(cmd, shell = True)