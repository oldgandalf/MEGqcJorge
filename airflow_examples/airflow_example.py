from airflow.decorators import dag, task
import pendulum
from airflow.operators.python import get_current_context
import mne


def get_var(key):
    ctx = get_current_context()
    val = ctx["dag_run"].conf[key]
    return val


@task()
def apply_mne_filters():
    data_path = get_var("data_path")
    l_freq = get_var("l_freq")
    h_freq = get_var("h_freq")
    raw = mne.io.read_raw_fif(data_path, preload=True)
    filt_raw = raw.copy().filter(l_freq, h_freq, fir_design='firwin')
    return filt_raw

#hardcoded parameters for filtering and path to data
#data_path = '/data2/egapontseva/MEG_QC_stuff/data/from openneuro/ds003483/sub-009/ses-1/meg/sub-009_ses-1_task-deduction_run-1_meg.fif'
#l_freq = 0.5
#h_freq = 100


@dag(start_date=pendulum.now())
def filtering():
    data_filtered = apply_mne_filters()

dag = filtering()

if __name__ == "__main__":
    dag.test(run_conf={"data_path": "/data2/egapontseva/MEG_QC_stuff/data/from_openneuro/ds003483/sub-009/ses-1/meg/sub-009_ses-1_task-deduction_run-1_meg.fif",
                       "l_freq" : 0.5,
                       "h_freq" : 100})


