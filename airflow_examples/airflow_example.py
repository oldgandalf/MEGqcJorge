from airflow.decorators import dag, task
import pendulum
from airflow.operators.python import get_current_context
import mne


def get_var(key):
    ctx = get_current_context()
    val = ctx["dag_run"].conf[key]
    return val


@task()
def load_meg_data(data_path):
    raw = mne.io.read_raw_fif(data_path, preload=True)
    return raw

@task()
def apply_mne_filters(raw, l_freq, h_freq):
    filt_raw = raw.copy().filter(l_freq, h_freq, fir_design='firwin')
    return filt_raw


@dag(start_date=pendulum.now())
def nilearn_fla():
    subjects = load_subjects()
    process_subject.expand(sub_label=subjects)
