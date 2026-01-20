'''Simple script that checks the status of the jobs submitted by runner on condor.

The status of the jobs can be checked by looking at the file in the jobs folder.

- job_x.idle: The job is waiting to be executed
- job_x.running: The job is running
- job_x.done: The job has finished
- job_x.failed: The job has failed
    
    where x is the job id.
'''

import os
import sys
import click
import glob
import subprocess as sp
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import time
import re


def get_tables(tot_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs, details=False):
    # Summary table
    table1 = Table(title="Job Summary")
    table1.add_column("Total jobs", style="cyan", no_wrap=True)
    table1.add_column("Idle jobs", style="blue", no_wrap=True)
    table1.add_column("Running jobs", style="magenta", no_wrap=True)
    table1.add_column("Done jobs", style="green", no_wrap=True)
    table1.add_column("Failed jobs", style="red", no_wrap=True)
    table1.add_row(str(len(tot_jobs)),
                  str(len(idle_jobs)),
                  str(len(running_jobs)),
                  str(len(done_jobs)),
                  str(len(failed_jobs)))
    # Create a table to display the status
    if details:
        table2 = Table(title="Job Status")
        table2.add_column("Job ID", style="cyan", no_wrap=True)
        table2.add_column("Submitted", style="blue", no_wrap=True)
        table2.add_column("Running", style="magenta", no_wrap=True)
        table2.add_column("Done", style="green", no_wrap=True)
        table2.add_column("Failed", style="red", no_wrap=True)
        for job in tot_jobs:
            table2.add_row(job,
                          "X" if job in idle_jobs else "",
                          "X" if job in running_jobs else "",
                          "X" if job in done_jobs else "",
                          "X" if job in failed_jobs else "")
    else:
        table2 = None
    return table1, table2


# Layout setup
def create_layout():
    layout = Layout()

    layout.split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    #layout["left"].size = 60  # You can adjust the width of the left panel
    return layout

def check_jobs_logs(jobs_folder):
     # Idle jobs
    idle_jobs = [ a.split("/")[-1][:-5] for a in glob.glob(f"{jobs_folder}/job_*.idle")]
    # Running jobs
    running_jobs = [a.split("/")[-1][:-8] for a in glob.glob(f"{jobs_folder}/job_*.running")]
    # Done jobs
    done_jobs = [ a.split("/")[-1][:-5] for a in glob.glob(f"{jobs_folder}/job_*.done")]
    # Failed jobs
    failed_jobs = [ a.split("/")[-1][:-7] for a in glob.glob(f"{jobs_folder}/job_*.failed")]
    return idle_jobs, running_jobs, done_jobs, failed_jobs


@click.command()
@click.option("-j", "--jobs-folder", type=str, help="Folder containing the jobs", required=True)
@click.option("-d","--details", is_flag=True, help="Show the details of the jobs")
@click.option("-r","--resubmit", is_flag=True, help="Resubmit the failed jobs")
@click.option("--max-resubmit", type=int, help="Maximum number of resubmission", default=3)
@click.option("--set-to-fail", is_flag=True, help="Set all jobs to failed. Use with caution!")
@click.option("--sub-replace", type=str, help="Replace this key's value with the new value", default=None)
@click.option("--skip-bad-files", is_flag=True, help="Add skip bad files option to condor_submit")



def check_jobs(jobs_folder, details, resubmit, max_resubmit, set_to_fail, sub_replace, skip_bad_files):
    
    jobs_folder = Path(jobs_folder)
    # Get the list of files in the folder
    tot_jobs = [ a.split("/")[-1][:-4] for a in glob.glob(f"{jobs_folder}/job_*.sub")]

    # Redo everything every 5 sec
    console = Console()

    failed_jobs_stats = {}
    tot_done = 0
    

    # Main loop
    layout = create_layout()
    idle_jobs, running_jobs, done_jobs, failed_jobs = check_jobs_logs(jobs_folder)
    tables = get_tables(tot_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs, details=details)
    layout["left"].update(Panel(tables[0], title="Job Status"))
    layout["right"].update(Panel("No logs yet", title="Log"))
    
    log_text = []
    definitive_failed = []
    
    with Live(layout, refresh_per_second=1/5, console=console):  # Refresh rate
        try:
            while True:
                idle_jobs, running_jobs, done_jobs, failed_jobs = check_jobs_logs(jobs_folder)

                if set_to_fail:
                    # Set all jobs to failed
                    for job in idle_jobs:
                        os.system(f"mv {jobs_folder}/{job}.idle {jobs_folder}/{job}.failed" )
                        failed_jobs.append(job)
                        idle_jobs.remove(job)
                    for job in running_jobs:
                        os.system(f"mv {jobs_folder}/{job}.running {jobs_folder}/{job}.failed")
                        failed_jobs.append(job)
                        running_jobs.remove(job)


                tables = get_tables(tot_jobs, idle_jobs, running_jobs, done_jobs, failed_jobs, details=details)
                # Update the left panel with a new table
                layout["left"].update(Panel(tables[0], title="Job Status"))

                # Checking failed jobs
                if len(failed_jobs) > 0:
                    if len(failed_jobs) > len(definitive_failed):
                        log_text.append("[red]Failed jobs found. Check the details below. Use --resubmit to resubmit the failed jobs[/]")
                    for failed_job in failed_jobs:
                        if failed_job in failed_jobs_stats:
                            if failed_job not in definitive_failed:
                                failed_jobs_stats[failed_job] += 1
                        else:
                            failed_jobs_stats[failed_job] = 1

                        if not failed_job in definitive_failed:
                            # Check the log file
                            glob_file = glob.glob(f"{jobs_folder}/logs/job_*.{failed_job.split('_')[1]}.err")
                            if glob_file:
                                with open(glob_file[-1]) as f:
                                    c = f.readlines()
                                    log_text.append( f"[b]Job {failed_job} failed[/] {failed_jobs_stats[failed_job]} times. Last error:")
                                    log_text.append("\t"+ "".join(c[-3:]))
                            else:
                                log_text.append( f"Error in job {failed_job}: No log file found")

                            if resubmit and failed_jobs_stats[failed_job] <= max_resubmit:
                                                                  
                                sub_file = f"{jobs_folder}/{failed_job}.sub"
                                if sub_replace is not None:

                                    sub_replace_dict = {i.split(":")[0]:i.split(":")[1] for i in sub_replace.split(",")}

                                    with open(sub_file) as f:
                                        lines = f.readlines()
                                    with open(sub_file, "w") as f:
                                        for line in lines:
                                            replace_line = False
                                            for key, value in sub_replace_dict.items():
                                                if line.startswith(key):
                                                    replace_line = True
                                                    print(f"Replacing {key} in {sub_file} with {value}")
                                                    f.write(line.replace(line.rstrip().split("=")[-1], value))
                                            if not replace_line:
                                                f.write(line)

                                # Check the proxy
                                job_file = f"{jobs_folder}/job.sh"
                                with open(job_file) as f:
                                    lines = f.readlines()
                                    for line in lines:
                                        if "X509_USER_PROXY" in line:
                                            proxy_path = line.split("=")[-1].strip()
                                            if not os.path.exists(proxy_path):
                                                print(f"Proxy file {proxy_path} does not exist. Reset by runnning")
                                                print(f"voms-proxy-init -voms cms --valid 192:00 -out {proxy_path}")
                                                exit(1)
                                            continue

                                if skip_bad_files:
                                    with open(job_file) as f:
                                        lines = f.readlines()
                                    with open(job_file, "w") as f:
                                        for line in lines:
                                            if "pocket-coffea run" in line and "--skip-bad-files" not in line:
                                                f.write(f"{line.rstrip()} --skip-bad-files\n")
                                            else:
                                                f.write(line)
                                   

                                os.system(f"rm {jobs_folder}/{failed_job}.failed")
                                os.system(f"touch {jobs_folder}/{failed_job}.idle")
                                os.system(f"cd {jobs_folder} && condor_submit {failed_job}.sub")
                            else:
                                # Add it to the list of jobs that are definitely failed
                                definitive_failed.append(failed_job)


                if len(log_text):
                    if len(log_text) > 20:
                        log_text = log_text[-20:]
                    layout["right"].update(Panel("\n".join(log_text), title="Log"))

                if len(tot_jobs) == len(done_jobs) + len(failed_jobs):
                    rprint("[green]All jobs are completed[/]")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            pass


check_jobs()