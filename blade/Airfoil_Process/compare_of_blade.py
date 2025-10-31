import numpy as np
import os,sys,importlib
import warnings
import shutil
import argparse
import sqlite3

import ruamel.yaml as yaml
yml = yaml.YAML(typ='rt', pure=True)

import helpers as util
importlib.reload(sys.modules['helpers'])


def main():

    parser = argparse.ArgumentParser(description="Plot exawind timeseries")

    parser.add_argument(
        "-i",
        "--infile",
        help="Input YAML file (must be present in the current directory)",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    with open(args.infile, 'r') as stream:
        loadyaml = yml.load(stream)

    fig,ax = util.setup_summary_plot(2,1,8,10,loadyaml)
    fig_l1,ax_l1 = util.setup_log_plot(1,1,8,5,"Solver Residuals")

    for i,case in enumerate(loadyaml['cases']):

        lab = case['label']
        parent = case['path_parent'] 
        bladepath = os.path.join(parent,case['blade_relpath'])
        sqlpath = os.path.join(parent,case['sql_relpath'])
        outpath =  os.path.join(parent,case['outfile_relpath'])

        bld_data = util.read_of_bladefile(bladepath)

        twist = np.array(bld_data.BlTwist)
        chord = np.array(bld_data.BlChord)
        span = np.array(bld_data.BlSpn)
        ax[0].plot(span,twist,label=lab)
        ax[1].plot(span,chord,label=lab)

        # Read log database
        conn_sql = sqlite3.connect(sqlpath)
        print(conn_sql)
        cursor = conn_sql.cursor()
        
        #cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        #tables = cursor.fetchall()
        #print(tables)

        #cursor.execute("SELECT * FROM solver_iterations LIMIT 1")
        #columns = [description[0] for description in cursor.description]
        #print("Column names:", columns)

        cursor.execute("SELECT solver_residuals FROM solver_iterations")
        log_timestamp = cursor.fetchall()
        print([row[0] for row in log_timestamp])


    ax[1].legend(loc='upper center', bbox_to_anchor=(0.8, 0.9),fancybox=False, shadow=False, ncol=1)
    plotpath = os.path.join(parent + '/blade_summary.png')
    fig.tight_layout()
    fig.savefig(plotpath)
        

if __name__ == "__main__":
    main()