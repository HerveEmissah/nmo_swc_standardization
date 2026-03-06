"""
Filename: app.py
Author: Herve Emissah
Created: 2021-08-15
Description: Flask-based backend service for SWC QC automation and ML model.
"""

from flask import Flask, request, jsonify, Response, send_file, send_from_directory, render_template, stream_with_context
from flask_cors import CORS
from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
from selenium.webdriver.firefox.options import Options
from werkzeug.utils import secure_filename
import shutil, os, traceback
import numpy as np
import subprocess
import time
import logging
import io
import zipfile
import datetime
import requests
import math

import networkx as nx
import pandas as pd
from scipy.spatial.distance import euclidean
from flask_rangerequest import RangeRequest
from threading import Lock
from typing import Set, Dict, Any, Optional, List, Tuple

import sys
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import normalize
from scipy.spatial import KDTree

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set Flask to run in production mode
os.environ['FLASK_ENV'] = 'production'

# Disable Flask's default logger
app.logger.disabled = True

# Create own logger
logger = logging.getLogger(__name__)

# Set the logger's level to ERROR to suppress all WARNINGS
logger.setLevel(logging.ERROR)

process_lock = Lock()
process_running = False

UPLOAD_FOLDER = 'Source-Version'  # Folder to store uploaded files
INVALID_FOLDER = 'invalid-swc' # Folder to filter out invalid swc files
DOWNLOAD_FOLDER = 'downloads'  # Folder to serve downloadable files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

ALLOWED_NODE_TYPES = {1, 2, 3, 4, 5, 6, 7}
TAG_SET = {5, 6, 7}
TEMP_TAG = 3

# clear log content if exists
file_path = '/nmo_swc/log/app.log'
if os.path.exists(file_path):
  # Open the file in write mode, which clears its contents
  with open(file_path, 'w') as file:
    pass

# Defining the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

@app.route("/nmo/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/nmo/SWC_STD', methods=["POST"])
def SWC_STD():
    
    if not start_process():
        return jsonify({"error": "Another process is already running"}), 429

    start_time = time.time()

    try:
        log_file = '/nmo_swc/log/app.log'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            pass

        # ---- ABSOLUTE PATHS ----
        upload_dir = '/nmo_swc/Source-Version'
        invalid_dir = '/nmo_swc/invalid-swc'

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(invalid_dir, exist_ok=True)
        os.chmod(upload_dir, 0o777)
        os.chmod(invalid_dir, 0o777)

        # Clear INVALID_FOLDER
        for f in os.listdir(invalid_dir):
            path = os.path.join(invalid_dir, f)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
                
        # ---- GET FILES FROM DISK ----
        uploaded_files = glob.glob(os.path.join(upload_dir, '*'))
        print("", flush=True)
        print("Num Files:", len(uploaded_files), flush=True)
        
        if not uploaded_files:
            return jsonify({
                "status": "error",
                "message": "No files found in upload directory."
            }), 400
        
        # ---- FILTER / VALIDATE FILES ----
        swc_files = []
        for src_path in uploaded_files:
            
            if not os.path.isfile(src_path):
                continue

            fname = os.path.basename(src_path)

            if not fname.lower().endswith('.swc'):
                print(f"********{fname} is invalid swc file")
                shutil.move(src_path, os.path.join(invalid_dir, fname))
                continue

            if not is_valid_swc(src_path):
                print(f"********{fname} has invalid (format/structure)")
                shutil.move(src_path, os.path.join(invalid_dir, fname))
                continue

            if not has_only_allowed_node_types(src_path):
                print(f"********{fname} has invalid node_type : (not in 1..7)")
                shutil.move(src_path, os.path.join(INVALID_FOLDER, fname))
                continue

            swc_files.append(src_path)

        if not swc_files:
            return jsonify({
                "error_code": 404,
                "message": "No valid .swc files found in the upload directory."
            }), 404


        # Extract the checkbox value and dropdown list value for Branch Tag Correction
        checked_correct_branch_tag = request.form.get('checkCorrectBranchTag') == 'true'
        branch_type = int(request.form.get('branchtype'))

        # Log the received values for debugging
        #print(f"Checked correct branch type: {checked_correct_branch_tag}")
        #print(f"branch type: {branch_type}")

        if checked_correct_branch_tag:
            print('\nCORRECT BRANCH TAG CHECKED...')
            print(f"NEW BRANCH TYPE IS: {branch_type}\n")
            update_all_branch_types(swc_files, branch_type)
  
        result = SWC_LONG_CONNECTIONS_STD()

        if isinstance(result, dict) and result.get("error_code") == 404:
            print("No .swc files found in the upload directory.")
            return jsonify(result), 404

        print('\n***Starting Measurements...')

        # Clear Archives2process folder
        os.system('rm -rf /nmo_swc/Archives2process/*')

        # Clear output_Final
        os.system('rm -rf /nmo_swc/output_Final/*')

        # Create measurement directories
        Measurement_dir = f'/nmo_swc/Archives2process/{archive_folder_name}'
        os.makedirs(Measurement_dir, exist_ok=True)
        os.chmod(Measurement_dir, 0o777)

        CNGVersion_dir = f'{Measurement_dir}/CNGVersion'
        os.makedirs(CNGVersion_dir, exist_ok=True)
        os.chmod(CNGVersion_dir, 0o777)

        Measurements_dir = f'{Measurement_dir}/Measurements'
        os.makedirs(Measurements_dir, exist_ok=True)
        os.chmod(Measurements_dir, 0o777)

        # IMPORTANT: Ensure CNG_dir exists before listing
        CNG_dir = '/nmo_swc/out_Final/CNG_Version/'
        if not os.path.exists(CNG_dir):
            return jsonify({
                "status": "error",
                "message": f"Expected folder not found: {CNG_dir}. SWC_LONG_CONNECTIONS_STD may have failed to create outputs."
            }), 500

        # Remove 0-byte files
        for filename in os.listdir(CNG_dir):
            fp = os.path.join(CNG_dir, filename)
            if os.path.isfile(fp) and os.stat(fp).st_size == 0:
                print(f"Not able to Process file: {filename}")
                os.remove(fp)

        # process files only if CNG_Version dir is not empty
        if not os.listdir(CNG_dir):
            out_dir = '/nmo_swc/out_Final/'
            os.chdir(out_dir)
            os.system("mv CNG_Version 'CNG Version'")

            output_dir = f'/nmo_swc/output_Final/{archive_folder_name}_Final'
            os.makedirs(output_dir, exist_ok=True)
            os.chmod(output_dir, 0o777)
        else:
            os.system(f'cp /nmo_swc/out_Final/CNG_Version/*.swc {CNGVersion_dir}/')
            os.system(f'chmod 777 {CNGVersion_dir}/*.*')

            working_dir = '/nmo_swc'
            os.chdir(working_dir)

            os.system('sh Run_LMProcess.sh')

            os.system(f'cp {Measurement_dir}/Measurements/*.* /nmo_swc/out_Final/Measurements/')
            os.system(f'rm -rf {Measurement_dir}/Measurements')
            os.system(f'rm -rf {Measurement_dir}/CNGVersion/*')

            os.system('cp /nmo_swc/Archives2process/*.txt /nmo_swc/out_Final/')
            os.system('rm -rf /nmo_swc/Archives2process/*.txt')

            # PNG automation
            print('\n***PNG Automation In Progress...')
            PNG_dir = '/nmo_swc/PNG_Automation_SN/PNG_Automation_SN'
            os.chdir(PNG_dir)
            os.system(f'rm -rf {PNG_dir}/PNG/*.*')
            os.system(f'rm -rf {PNG_dir}/SWC/*.*')
            os.system(f'cp /nmo_swc/out_Final/CNG_Version/*.swc {PNG_dir}/SWC/')
            os.system('sh /nmo_swc/PNG_Automation_SN/PNG_Automation_SN/PNG_generator.sh')

            Img_dir = '/nmo_swc/out_Final/Images'
            os.makedirs(Img_dir, exist_ok=True)
            os.chmod(Img_dir, 0o777)

            os.system('mkdir -p /nmo_swc/out_Final/Images/PNG')
            png_files = glob.glob(os.path.join(PNG_dir, 'PNG', '*.png'))
            if png_files:
               os.system(f'cp {PNG_dir}/PNG/*.png /nmo_swc/out_Final/Images/PNG/')
               os.system(f'rm -rf {PNG_dir}/PNG/*.png')

            # rename folders in out_Final
            out_dir = '/nmo_swc/out_Final/'
            os.chdir(out_dir)
            os.system("mv CNG_Version 'CNG Version'")
            os.system("mv Remaining_issues 'Remaining issues'")
            os.system("mv /nmo_swc/Standardizationlog 'Standardization log'")

            # Copy all files from out_Final to mounted output_Final
            os.system(f"mkdir -p /nmo_swc/output_Final/{archive_folder_name}_Final")
            output_dir = f'/nmo_swc/output_Final/{archive_folder_name}_Final'
            os.chmod(output_dir, 0o777)
            os.system(f'cp -r /nmo_swc/out_Final/* /nmo_swc/output_Final/{archive_folder_name}_Final/')

            # Move .std files
            os.system(f"mkdir -p /nmo_swc/output_Final/{archive_folder_name}_Final/Possible-issues")
            os.system('chmod 777 /nmo_swc/Normalized/Possible-issues/*.std')
            os.system(f'cp /nmo_swc/Normalized/Possible-issues/*.std /nmo_swc/output_Final/{archive_folder_name}_Final/Possible-issues/')

            # Move logs
            log1 = f"/nmo_swc/output_Final/{archive_folder_name}_Final/Log1.txt"
            log2 = f"/nmo_swc/output_Final/{archive_folder_name}_Final/Log2.txt"
            os.system(f"mv /nmo_swc/Normalized/Possible-issues/Log.txt {log1}")
            os.system(f"mv /nmo_swc/output_Final/{archive_folder_name}_Final/'Remaining issues'/Log.txt {log2}")
            if os.path.exists(log1): os.chmod(log1, 0o777)
            if os.path.exists(log2): os.chmod(log2, 0o777)

        # Copy all original files to Source-Version inside output
        os.system(f"mkdir -p /nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version")
        os.system(f"cp {upload_dir}/* /nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version/")
        invalid_files_glob = glob.glob(os.path.join(invalid_dir, "*"))
        #if invalid_files_glob:
        #    os.system(f"cp {invalid_dir}/* /nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version/")

        if invalid_files_glob:
           os.system(f"cp {invalid_dir}/* " 
                     f"/nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version/"
           )

        # Control file creation
        source_version_dir = f'/nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version'
        cng_version_dir = os.path.join('/nmo_swc/output_Final', f'{archive_folder_name}_Final', 'CNG Version')
        control_file = f'/nmo_swc/output_Final/{archive_folder_name}_Final/control.txt'

        os.makedirs(os.path.dirname(control_file), exist_ok=True)
        if os.path.exists(control_file):
            os.remove(control_file)

        source_files = get_files_list(source_version_dir)
        cng_files = get_files_list(cng_version_dir)

        source_files_without_ext = []
        for file in source_files:
            if file.endswith('.swc'):
                try:
                    source_files_without_ext.append(get_filename_without_extension(file))
                except Exception as e:
                    print(f"Error getting source file: {file} - {e}")

        cng_files_without_ext = []
        for file in cng_files:
            if file.endswith('.CNG.swc'):
                try:
                    cng_files_without_ext.append(get_filename_without_extension(file))
                except Exception as e:
                    print(f"Error with processing CNG file: {file} - {e}")

        source_bases = {base_name(f) for f in source_files}
        cng_bases = {base_name(f) for f in cng_files}

        #missing_files = [f for f in source_files_without_ext if f not in cng_files_without_ext]
        missing_files = [
            f for f in source_files
            if base_name(f) not in cng_bases
        ]

        png_control_file = "/nmo_swc/png_control.txt"

        invalid_files = [os.path.basename(f) for f in invalid_files_glob]
        with open(control_file, "w") as f:
            f.write(f"Total swc files in Source-version folder: {len(source_files)}\n")
            f.write(f"Total processed swc files in CNG-Version folder: {len(cng_files)}\n\n")
            elapsed = time.time() - start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            f.write(f"Total processing time: {h:02d}:{m:02d}:{s:02d} ({int(elapsed)} seconds)\n\n")

            f.write("swc Files in Source-version not processed:\n")
            for name in missing_files:
                f.write(f"{name}\n")

            #f.write("\nInvalid swc Files:\n")
            #missing_bases = {base_name(f) for f in missing_files}
            #for name in invalid_files:
            #    if base_name(name) not in missing_bases:
            #        f.write(f"{name}\n")

            f.write("\n")  # spacing before PNG failures

            # ---- APPEND PNG CONTROL CONTENT ----
            if os.path.exists(png_control_file):
                f.write("PNG generation failures:\n")

                with open(png_control_file, "r") as pf:
                    for line in pf:
                        f.write(line)

        # permissions
        os.chmod(control_file, 0o777)

        # ---- DELETE PNG CONTROL FILE AFTER APPEND ----
        if os.path.exists(png_control_file):
            os.remove(png_control_file)

        os.system("chmod -R 777 /nmo_swc/output_Final/")

        print('\n**SWC STANDARDIZATION COMPLETED**')

        # Clear upload_dir after processing (optional)
        clear_folder(upload_dir)

        return jsonify({
            "status": "success",
            "message": "SWC Standardization Completed Successfully",
            "uploaded_files": len(swc_files)
        }), 200

    except Exception as e:
        print("STANDARDIZATION FAILED:", str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        end_process()

def base_name(fname):
    if fname.endswith(".CNG.swc"):
        return fname[:-8]
    return os.path.splitext(fname)[0]


def update_all_branch_types(swc_files, new_branch_type):
    """
    Update ALL node types (2nd column) in each SWC file to new_branch_type.
    Overwrites files in place.
    """
    new_branch_type = str(int(new_branch_type))

    for path in swc_files:
        with open(path, "r") as f:
            lines = f.readlines()

        out_lines = []

        for line in lines:
            stripped = line.strip()

            # keep blank lines and comments
            if not stripped or stripped.startswith("#"):
                out_lines.append(line)
                continue

            # handle inline comments
            main, hash_, comment = line.partition("#")
            parts = main.split()

            # expect at least 7 SWC columns
            if len(parts) < 7:
                out_lines.append(line)
                continue

            # SWC format: n T x y z r parent
            parts[1] = new_branch_type

            new_line = " ".join(parts)
            if hash_:
                new_line += "  #" + comment.strip()

            out_lines.append(new_line + "\n")

        # overwrite file
        with open(path, "w") as f:
            f.writelines(out_lines)


def is_valid_swc(filepath):
    nodes = set()

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                # SWC must have exactly 7 columns
                if len(parts) != 7:
                    return False

                # Parse values
                node_id = int(parts[0])
                node_type = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                radius = float(parts[5])
                parent = int(parts[6])

                nodes.add(node_id)

                # Parent must be -1 or already defined
                if parent != -1 and parent not in nodes:
                    return False

        return len(nodes) > 0

    except Exception:
        return False

# --- Check and start Tomcat if needed ---
def is_tomcat_running(url="http://localhost:8080", timeout=2):
    """
    Returns True if Tomcat responds to HTTP request.
    Accepts 200 or 404 as valid.
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code in (200, 404)
    except requests.RequestException:
        return False

# Function to set read, write, and execute permissions for all users
def set_permissions_recursively(directory):
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.chmod(dir_path, 0o777)
        for name in files:
            file_path = os.path.join(root, name)
            os.chmod(file_path, 0o777)

# Function to get list of files in a directory
def get_files_list(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            files_list.append(file)
    return files_list

def get_filename_without_extension(filename):
    # Find the index of the first period in the filename
    first_period_index = filename.find('.')
    if first_period_index != -1:  # If a period is found
        # Return the filename before the first encountered period
        return filename[:first_period_index]
    else:
        # If no period is found, return the original filename
        return filename

def find_files_with_code(filename, code):
    working_dir = '/nmo_swc/nmo_user_Final/Remaining_issues'
    os.chdir(working_dir)
    files_with_code = []
    with open(filename, 'r') as file:
        current_file = None
        for line in file:
            if line.strip().endswith(".swc.std"):
                current_file = line.strip()
            elif code in line.strip() and current_file:
                files_with_code.append(current_file)
    return files_with_code

def SWC_Fix_Zero_Radius(inputFilename):

    working_dir = '/nmo_swc/nmo_user_Final/CNG_Version'
    os.chdir(working_dir)

    outputFilename = "qc_" + inputFilename
    t = {}
    x = {}
    y = {}
    z = {}
    r = {}
    p = {}
    count = 0

    try:
        with open(inputFilename, 'r') as input_file, open(outputFilename, 'w') as output_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.startswith("#"):
                    #print(line)
                    fields = line.split()
                    if len(fields) >= 7:
                        try:
                            index = int(fields[0])
                            t[index] = int(fields[1])
                            x[index] = float(fields[2])
                            y[index] = float(fields[3])
                            z[index] = float(fields[4])
                            r[index] = float(fields[5])
                            p[index] = float(fields[6])
                            count += 1
                        except ValueError:
                            pass #print(f"Error: Line {line_number} has invalid data: {line.strip()}", file=sys.stderr)
                    else:
                        pass #print(f"Error: Line {line_number} does not have enough fields: {line.strip()}", file=sys.stderr)

            for i in range(1, count + 1):
                if i in r and r[i] <= 0:
                    r[i] = 0.5
                if i in t and t[i] == 0:
                    t[i] = 3
                if i in t and i in x and i in y and i in z and i in r and i in p:
                    print(i, t[i], x[i], y[i], z[i], r[i], p[i], file=output_file)

        input_file.close()
        output_file.close()

        # Delete the input file
        os.remove(inputFilename)

        # Rename the output file with the input file name
        os.rename(outputFilename, inputFilename)

        os.chmod(inputFilename, 0o777)

    except FileNotFoundError:
        print(f"Error: File '{inputFilename}' not found.", file=sys.stderr)
    except IOError:
        print(f"Error: Unable to read or write file '{inputFilename}'.", file=sys.stderr)


@app.route('/nmo/upload', methods=['POST'])
def upload():
    global archive_folder_name
    if not start_process():
        return jsonify({"error": "Another process is already running"}), 429

    try:
        if 'files' not in request.files:
            print('No files uploaded')
            return jsonify({'message': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        if not files:
            print('***No files uploaded***')
            return jsonify({'message': 'No files uploaded'}), 400

        folder_path = files[0].filename.rsplit('/', 1)[0]
        archive_folder_name = folder_path.rsplit('/', 1)[-1].replace(' ', '')
        print('ARCHIVE NAME:' + archive_folder_name)

        # clear log content if exists
        file_path = '/nmo_swc/log/app.log'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            pass

        upload_folder_path = f'/nmo_swc/{UPLOAD_FOLDER}'
        empty_folder(upload_folder_path)

        curr_wrkdir = '/nmo_swc'
        os.chdir(curr_wrkdir)

        saved_files = []

        for file in files:
            filename_only = os.path.basename(file.filename)
            print('Saving ' + filename_only)
            dst_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_only)
            file.save(dst_path)
            os.chmod(dst_path, 0o777)

            print(filename_only + ' uploaded and saved in CNG Server\n')
            saved_files.append(filename_only)

        return jsonify({
            "status": "success",
            "message": "Files uploaded successfully",
            "archive_folder_name": archive_folder_name,
            "saved_files": saved_files,
            "saved_count": len(saved_files)
        }), 200

    except Exception as e:
        print("FILES UPLOAD FAILED:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        end_process()

@app.route('/nmo/download', methods=['GET'])
def download():
    try:
        print('\n\n***Downloading...')
        timestamp = request.args.get('timestamp', '')

        # Mounted container output folder
        output_folder = '/nmo_swc/output_Final'
        #output_folder = '/nmo_swc/out_Final'
        zip_filename = f'swc_standardized_{timestamp}.zip'
        zip_path = os.path.join(output_folder, zip_filename)

        # Check if the output folder is empty
        if not os.listdir(output_folder):
            print('No file to download')
            return 'Output folder is empty', 400

        # Create a ZIP archive or update an existing one
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            # Add all files in the output folder to the archive
            for root, dirs, files in os.walk(output_folder):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Skip .swp files and already compressed files like .zip
                    if not file.endswith('.swp') and not file.endswith('.zip'):
                        # Check file size (Include files of all sizes)
                        if os.path.getsize(file_path) > 0 or not file.endswith('.zip'):
                            archive.write(file_path, os.path.relpath(file_path, output_folder))
                            #print(f'Added {file_path} to the archive')
                        else:
                            print(f'Skipped {file_path} (zero size compressed file)')

        # Send the ZIP archive as a download with a custom filename
        response = send_file(
            zip_path,
            as_attachment=True,
            mimetype='application/zip'
        )

        # Log a download status message
        time.sleep(5)
        print('\n**DOWNLOAD COMPLETED**')
        print(' ')

        return response
    except Exception as e:
        # Log an error message if an exception occurs
        print(f'Error during download: {str(e)}')
        return 'Error during download', 500


def create_multi_zip(input_folder, output_folder, max_size_per_zip=200 * 1024 * 1024):
    zip_files = []
    current_files = []
    current_size = 0
    zip_counter = 1

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.swc'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)

                # If the file size exceeds the maximum size per ZIP file, create a separate ZIP file
                if file_size > max_size_per_zip:
                    with zipfile.ZipFile(os.path.join(output_folder, f'{file}_{zip_counter}.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
                        archive.write(file_path, arcname=file)
                    zip_counter += 1
                    continue

                # If adding this file exceeds the maximum size per ZIP file, create a new ZIP file
                if current_size + file_size > max_size_per_zip:
                    zip_filename = os.path.join(output_folder, f'swc_{zip_counter}.zip')
                    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as archive:
                        for file_to_add in current_files:
                            archive.write(file_to_add, arcname=os.path.relpath(file_to_add, input_folder))
                    zip_files.append(zip_filename)
                    zip_counter += 1
                    current_files = []
                    current_size = 0

                current_files.append(file_path)
                current_size += file_size

    # Create the last ZIP file with the remaining files
    if current_files:
        zip_filename = os.path.join(output_folder, f'swc_{zip_counter}.zip')
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file_to_add in current_files:
                archive.write(file_to_add, arcname=os.path.relpath(file_to_add, input_folder))
        zip_files.append(zip_filename)

    return zip_files

def create_multi_zip2(input_folder, output_folder, max_size_per_zip=200 * 1024 * 1024):
    zip_files = []
    current_files = []
    current_size = 0
    zip_counter = 1

    # Walk through the input folder to gather all files
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)

            # If the file size exceeds the maximum size per ZIP file, create a separate ZIP file for it
            if file_size > max_size_per_zip:
                with zipfile.ZipFile(os.path.join(output_folder, f'{file}_{zip_counter}.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
                    archive.write(file_path, arcname=file)
                zip_counter += 1
                continue

            # If adding this file exceeds the maximum size per ZIP file, create a new ZIP file
            if current_size + file_size > max_size_per_zip:
                zip_filename = os.path.join(output_folder, f'archive_{zip_counter}.zip')
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for file_to_add in current_files:
                        archive.write(file_to_add, arcname=os.path.relpath(file_to_add, input_folder))
                zip_files.append(zip_filename)
                zip_counter += 1
                current_files = []
                current_size = 0

            # Add the current file to the list
            current_files.append(file_path)
            current_size += file_size

    # Create the last ZIP file with the remaining files
    if current_files:
        zip_filename = os.path.join(output_folder, f'archive_{zip_counter}.zip')
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file_to_add in current_files:
                archive.write(file_to_add, arcname=os.path.relpath(file_to_add, input_folder))
        zip_files.append(zip_filename)

    return zip_files

@app.route('/nmo/download_connected', methods=['GET'])
def download_connected():
    try:
        print('\n\n***Downloading...')
        timestamp = request.args.get('timestamp', '')

        # Mounted container output folder
        input_folder = '/nmo_swc/output'
        output_folder = '/nmo_swc/output_zips'
        max_size_per_zip = 200 * 1024 * 1024  # 200MB

        # Create multiple ZIP files with the specified logic
        zip_files = create_multi_zip2(input_folder, output_folder, max_size_per_zip)

        # Stream the ZIP files as a download with custom filenames
        response = Response(stream_with_context(generate_zips(zip_files)), mimetype='application/zip')
        response.headers['Content-Disposition'] = 'attachment; filename=swc_connected_zips.zip'

        # Log a download status message
        time.sleep(5)
        print('\n**DOWNLOAD COMPLETED**')
        print(' ')

        return response
    except Exception as e:
        # Log an error message if an exception occurs
        print(f'Error during download: {str(e)}')
        return 'Error during download', 500

def generate_zips(zip_files):
    for zip_file in zip_files:
        with open(zip_file, 'rb') as f:
            yield f.read()


@app.route('/nmo/download_corrected_tags', methods=['GET'])
def download_corrected_tags():
    try:
        print('\n\n***Downloading...')
        timestamp = request.args.get('timestamp', '')

        # Mounted container output folder
        input_folder = '/nmo_swc/output_Tag'
        output_folder = '/nmo_swc/output_Tag_zips'
        max_size_per_zip = 200 * 1024 * 1024  # 200MB

        # Create multiple ZIP files with the specified logic
        zip_files = create_multi_zip2(input_folder, output_folder, max_size_per_zip)

        # Stream the ZIP files as a download with custom filenames
        response = Response(stream_with_context(generate_zips(zip_files)), mimetype='application/zip')
        response.headers['Content-Disposition'] = 'attachment; filename=swc_auto_tag_zips.zip'

        # Log a download status message
        time.sleep(5)
        print('\n**DOWNLOAD COMPLETED**')
        print(' ')

        return response
    except Exception as e:
        # Log an error message if an exception occurs
        print(f'Error during download: {str(e)}')
        return 'Error during download', 500


@app.route('/nmo/readfile', methods=['GET'])
def readfile():
    file_path = '/nmo_swc/log/app.log'
    try:
        with open(file_path, 'r') as file:
            file_content = file.readlines()

        # Create a filtered list of lines, excluding those with the specified pattern
        filtered_lines = [line for line in file_content
                          if '* Running on http://0.0.0.0:5000/' not in line
                          and 'GET /nmo' not in line
                          and 'POST /nmo' not in line
                          and '* Serving Flask app' not in line
                          and '* Environment:' not in line
                          and 'WARNING:' not in line
                          and 'Use a ' not in line
                          and ' * Debug mode:' not in line
                         ]

        # Join the filtered lines to form the content
        filtered_content = ''.join(filtered_lines)

        return Response(filtered_content, content_type='text/plain')
    except FileNotFoundError:
        return 'File not found', 404

@app.route('/nmo/PNG_Automation_SN', methods=['GET'])
def PNG_Automation_SN():

    # Define the path to your SWC files
    PNG_directory = '/nmo_swc/PNG_Automation_SN/PNG_Automation_SN/'
    os.chdir(PNG_directory)
    os.system(f'rm -rf {PNG_directory}/SWC/*.*')
    os.system(f'cp /nmo_swc/out_Final/CNG_Version/*.swc {PNG_directory}/SWC/')
    os.system('sh PNG_generator.sh')

    return '****PNG_Automation completed'

@app.route('/nmo/Measurement_Extraction', methods=['GET'])
def Measurement_Extraction():

    # Define the path to your SWC files
    wrk_dir = '/nmo_swc/'
    os.chdir(wrk_dir)
    os.system('sh Run_LMProcess.sh')

    return '****Measurement_Extractiony completed'

@app.route('/nmo/convert_swc_to_png', methods=['GET'])
def convert_swc_to_png():

    # Define the path to the SWC files
    swc_directory = '/nmo_swc'

    # Set up Firefox options
    firefox_options = Options()
    firefox_options.add_argument('--headless')
    firefox_options.add_argument('--no-sandbox')
    firefox_options.add_argument('--disable-dev-shm-usage')

    #chrome_options.add_argument('--disable-gpu')
    #chrome_options.add_argument('--window-size=1920x1080')

    # Initialize the WebDriver with Firefox
    browser = webdriver.Firefox(options=firefox_options)

    try:
        # Construct the SWC file URL
        filename = 'A1_5.CNG.swc'
        swc_directory = '/nmo_swc'
        swc_url = f'file://localhost//{os.path.join(swc_directory, filename)}'

        # Navigate to the SWC file URL
        #param = f'http://cng-nmo-dev3.orc.gmu.edu:8080/swc/api/view?url={swc_url}&portable=true'
        param ='http://cng-nmo-dev3.orc.gmu.edu:8080/swc/api/view?url=https://neuromorpho.org/dableFiles/cardona/CNG version/A02m_a1l_morphology.CNG.swc&portable=true'
        browser.get(param)
        #browser.get(f'http://cng-nmo-dev3.orc.gmu.edu:8080/swc/api/view?url=file:///nmo_swc/A1_5.CNG.swc')

        # Capture screenshot and save as PNG
        screenshot_path = os.path.join(swc_directory, f'{filename}.png')
        browser.save_screenshot(screenshot_path)

        return param # f'Screenshot captured and saved as {screenshot_path}'

    finally:
        # Close the browser
        browser.quit()


# New route to clear the log
@app.route('/nmo/clearlog', methods=['POST'])
def clearlog():
    log_file_path = '/nmo_swc/log/app.log'
    try:
        with open(log_file_path, 'w') as log_file:
            log_file.write('Log cleared at ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        return 'Log cleared', 200
    except Exception as e:
        return 'Failed to clear log', 500


# Function to find the main tree and isolated trees
def identify_trees(G):
    # Find all connected components (trees)
    connected_components = list(nx.connected_components(G))

    # Find the soma node (parent == -1 and node_type == 1)
    soma_node = None
    for node in G.nodes:
        if G.nodes[node]['parent'] == -1 and G.nodes[node]['node_type'] == 1:
            soma_node = node
            break

    # If soma_node is None, consider the main_tree as the tree with node_type == 2
    if soma_node is None:
        main_tree = None
        for component in connected_components:
            for node in component:
                if G.nodes[node]['node_type'] == 2:
                    main_tree = component
                    break
            if main_tree:
                break

        # If no node_type == 2 found, check for node_type == 3
        if main_tree is None:
            for component in connected_components:
                for node in component:
                    if G.nodes[node]['node_type'] == 3:
                        main_tree = component
                        break
                if main_tree:
                    break
        
        # If no node_type == 3 found, check for node_type == 4
        if main_tree is None:
            for component in connected_components:
                for node in component:
                    if G.nodes[node]['node_type'] == 4:
                        main_tree = component
                        break
                if main_tree:
                    break

        if main_tree is None:
            raise ValueError("No soma node or node_type == 2, 3, or 4 found in the graph")
    else:
        # If soma_node is found, identify the main tree as the component containing soma_node
        main_tree = None
        for component in connected_components:
            if soma_node in component:
                main_tree = component
                break

    # Identify isolated trees: the components that don't contain the main tree
    isolated_trees = [comp for comp in connected_components if comp != main_tree]

    return main_tree, isolated_trees


# Function to find the soma node (node_type=1, parent=-1)
def find_soma(G):
    for n in G.nodes:
        if G.nodes[n].get('node_type') == 1:
            return n
    return list(G.nodes)[0]  # fallback if no soma

# Function to get positions of nodes (from 'pos' attribute)
def get_positions(G):
    positions = {}
    for node in G.nodes:
        if 'pos' in G.nodes[node]:  # Check if 'pos' exists
            positions[node] = G.nodes[node]['pos']  # Use 'pos' attribute directly
        else:
            print(f"Warning: Node {node} is missing position attribute ('pos'). Skipping this node.")
    
    return positions

def bfs_order_and_set_parents(G, root):
    visited = set()
    parent_map = {root: -1}
    queue = [root]

    while queue:
        current = queue.pop(0)
        visited.add(current)
        for neighbor in G.neighbors(current):
            if neighbor not in visited and neighbor not in queue:
                parent_map[neighbor] = current
                queue.append(neighbor)
    return parent_map

def assign_parents(G, parent_map):
    for node, parent in parent_map.items():
        G.nodes[node]['parent'] = parent

# Function to find the leaf nodes of a tree (nodes with degree 1)
def find_leaf_nodes(G, tree):
    # Leaf nodes have degree 1 (only connected to one node)
    leaf_nodes = [node for node in tree if G.degree(node) == 1]
    return leaf_nodes

# Function to connect a given isolated node to the soma node and re-root the tree
def connect_single_subtree(G, closest_end_node):
    # Identify the component of the closest_end_node and the main component (with soma)
    components = list(nx.connected_components(G))
    target_component = None
    for comp in components:
        if closest_end_node in comp:
            target_component = comp
            break

    # Soma must be in the main tree
    soma_node = find_soma(G)
    soma_component = [comp for comp in components if soma_node in comp][0]

    if target_component == soma_component:
        print("Target node is already connected to the soma.")
        return

    # Ensure soma node has a position, if not raise an error
    if 'pos' not in G.nodes[soma_node]:
        print(f"Error: Soma node {soma_node} is missing position information.")
        return

    # Get positions of all nodes
    pos = get_positions(G)

    # Ensure that the closest_end_node has a valid position
    if closest_end_node not in pos:
        print(f"Error: Node {closest_end_node} is missing position information.")
        return

    # Build KDTree of soma component positions
    soma_nodes = list(soma_component)
    kdtree = KDTree([pos[n] for n in soma_nodes])

    # Connect closest_end_node to soma
    G.add_edge(closest_end_node, soma_node)

    # Re-root entire graph from soma
    parent_map = bfs_order_and_set_parents(G, soma_node)
    assign_parents(G, parent_map)

@app.route('/nmo/connect_disjoint_subtrees', methods=["GET", "POST"])
def connect_disjoint_subtrees():

    if not start_process():
        return jsonify({"error": "Another process is already running"}), 429

    overall_start_time = time.time()

    try:    
       # clear log content if exists
       file_path = '/nmo_swc/log/app.log'
       if os.path.exists(file_path):
          # Open the file in write mode, which clears its contents
          with open(file_path, 'w') as file:
             pass

       # Extract the selected files from the form data
       selected_files = request.files.getlist('files')

       # Extract the checkbox value and dropdown list value
       check_long_connections = request.form.get('checkLongConnections') == 'true'
       stdev_x = int(request.form.get('stdevX'))

       # Log the received values for debugging
       #print(f"Check long connections: {check_long_connections}")
       #print(f"Stdev: {stdev_x}")

       # Set working directory to /nmo_swc
       working_dir = '/nmo_swc'
       os.chdir(working_dir)
       upload_dir = '/nmo_swc/Source-Version'
       SWC_dir = '/nmo_swc/SWC'
       output_dir = '/nmo_swc/output'
       lib = '/nmo_swc/plugins/neuron_utilities/neuron_connector/libneuron_connector.so'
       if not os.path.exists(output_dir):
         os.makedirs(output_dir)

       # Remove existing files in the output directory
       for swc_file in glob.glob(os.path.join(SWC_dir, '*.*')):
          os.remove(swc_file)

       for item in glob.glob(os.path.join(output_dir, '*')):
           os.remove(item)

       # Check if the upload directory is not empty
       swc_files = glob.glob(os.path.join(upload_dir, '*.swc'))
       if not swc_files:
           print()
           print('Please Upload .swc files to proceed...\n')
           return "No .swc files found in the upload directory."

       # Initialize a list to hold file data
       file_metrics = []
       
       # First check for side branches and overlapping points code
       print('CHECKING OVERLAPPING POINTS, SIDE BRANCHES AND MINIMUM RADIUS...')
       result = SWC_LONG_CONNECTIONS_STD()

       #move standardized file to directory for long connections processing
       long_connect_dir = f'/nmo_swc/long_connect'    
       if os.path.exists(long_connect_dir):
          subprocess.run(['sudo', 'chmod', '-R', '777', long_connect_dir], check=True)
          shutil.rmtree(long_connect_dir)
       os.makedirs(long_connect_dir, exist_ok=True)
       os.chmod(long_connect_dir, 0o777)

       source_dir = "/nmo_swc/out/"
       destination_dir = long_connect_dir

       # Loop through all files in the source_dir and remove those with size 0
       for filename in os.listdir(source_dir):
           file_path = os.path.join(source_dir, filename)    
           # Check if it is a file and has a size of 0 bytes
           if os.path.isfile(file_path) and os.stat(file_path).st_size == 0:
               file_name = os.path.basename(file_path)
               print(f"Not able to Process file: {file_name}")
               os.remove(file_path)
       
       # process the files only if source_dir is not empty 
       if not os.listdir(source_dir):
           print('No file to process.  source_dir is empty')

       else:
           # Copy files
           for file_path in glob.glob(f"{source_dir}*.swc"):
               shutil.copy(file_path, destination_dir)
        
           if check_long_connections:
               # Checking long connections
               print('\nCHECKING LONG CONNECTIONS...')
           long_connect_dir = f'/nmo_swc/long_connect'
           for swc_file in glob.glob(os.path.join(long_connect_dir, '*.swc')):
              # Initialize file_data to "N/A"
              file_data = {
                  'file_name': 'N/A',
                  'mean_distance': 'N/A',
                  'std_distance': 'N/A',
                  'threshold': 'N/A',
                  'long_connections': 'N/A',
                  'num_long_connections_removed': 'N/A',
                  'num_isolated_nodes': 'N/A'
              }

              swc_df = read_swc_pandas(swc_file)
              file_name = os.path.basename(swc_file)
              try:
                 G = create_graph_nx(swc_df)
              except Exception as e:
                 print(f"Error creating graph for file {swc_file}: {e}")
                 continue

              distances = []
              distance_pairs = []

              # Calculate and store distances between connected nodes
              for u, v in G.edges():
                  node_u = G.nodes[u]
                  node_v = G.nodes[v]
                  if 'pos' in node_u and 'pos' in node_v:
                      distance = euclidean_distance(node_u, node_v)
                      distances.append(distance)
                      distance_pairs.append((u, v, distance))
                      #print(f"Distance between node {u} and node {v}: {distance:.2f}")
                  else:
                      print(f"Missing 'pos' attribute in nodes {u} or {v}")

              if distances:
                  mean_distance = np.mean(distances)
                  std_distance = np.std(distances)
              
                  print(f"\nChecking Long Connections in file: {file_name}")
                  print(f"Mean distance: {mean_distance:.5f}")
                  print(f"StDev: {std_distance:.5f}")

                  edges_removed = False
                  # Print nodes and distances that are greater than 4 times the standard deviation
                  threshold = stdev_x * std_distance

                  if check_long_connections:
                      print(f"Based on distance greater than {stdev_x} times the StDev ({threshold:.2f}):")
                      total_connections = len(distance_pairs)
                      long_connections = 0
                      for u, v, distance in distance_pairs: 
                          node_u_type = G.nodes[u].get('node_type', None)
                          node_v_type = G.nodes[v].get('node_type', None)

                          # Skip processing if either node is soma (type 1)
                          #if node_u_type == 1 or node_v_type == 1:
                          #    continue
                       
                          if distance > threshold:
                              long_connections += 1
                              #print(f"- Long Connection found between nodes {u} and {v}: d = {distance:.2f}")
                              G.remove_edge(u, v)
                           
                              # If u is the parent of v, set parent of v to -1
                              if G.nodes[v]['parent'] == u:
                                  G.nodes[v]['parent'] = -1
                                  #print(f"Node {v} was child of {u}. Setting parent of {v} to -1.")

                              # If v is the parent of u, set parent of u to -1
                              elif G.nodes[u]['parent'] == v:
                                  G.nodes[u]['parent'] = -1
                                  #print(f"Node {u} was child of {v}. Setting parent of {u} to -1.")
                               
                              edges_removed = True
              
                      if long_connections == 1:
                         print(f"Found {long_connections} long connection out of {total_connections} total connections")
                      else:
                         print(f"Found {long_connections} long connections out of {total_connections} total connections")
         
                      if edges_removed:
                          # Get isolated nodes
                          isolated_nodes = list(nx.isolates(G))

                          # Filter out isolated nodes of type 1 (keep only nodes with node_type != 1)
                          nodes_to_remove = [node for node in isolated_nodes if G.nodes[node].get('node_type') != 1]
    
                          num_isolated_nodes = len(nodes_to_remove)
                          print(f"Number of isolated nodes (excluding node_type=1): {num_isolated_nodes}")

                          # Reassign parent to -1 for nodes whose parent is an isolated node
                          for node_id in G.nodes:
                              parent_id = G.nodes[node_id].get('parent', None)
                              if parent_id in isolated_nodes:
                                  G.nodes[node_id]['parent'] = -1

                          # Remove the isolated nodes that are not of type 1
                          G.remove_nodes_from(nodes_to_remove)

                          # Define output filename based on the input SWC file name
                          name, ext = os.path.splitext(swc_file)
                          output_filename = f'{name}.longfixed{ext}'

                          # Write the updated SWC file
                          write_swc(G, output_filename)

                          # Store the metrics for this file in a dictionary
                          file_data = {
                              'file_name': os.path.basename(output_filename),
                              'mean_distance': mean_distance,
                              'std_distance': std_distance,
                              'threshold': threshold,
                              'long_connections': long_connections,
                              'num_long_connections_removed': long_connections,
                              'num_isolated_nodes': num_isolated_nodes
                          }
                          # Append the data for this file to the list
                          file_metrics.append(file_data)
                          os.remove(swc_file)  # Remove the original file after creating new one
                  else:
                      name, ext = os.path.splitext(swc_file)
                      output_filename = f'{name}.long_unchecked{ext}'
                      write_swc(G, output_filename)

                      # Store the metrics for this file in a dictionary
                      file_data = {
                        'file_name': os.path.basename(output_filename),
                        'mean_distance': mean_distance,
                        'std_distance': std_distance,
                        'threshold': threshold,
                        'long_connections': 'N/A',
                        'num_long_connections_removed': 'N/A',
                        'num_isolated_nodes': 'N/A'
                      }

                      # Append the data for this file to the list
                      file_metrics.append(file_data)
                      os.remove(swc_file)  # Remove the original file after creating new one

           # Connect Disjoint Subtrees - Process each .swc file in the input directory
           print('\nCONNECTING DISJOINT SUBTREES...\n')    
           for swc_file in glob.glob(os.path.join(long_connect_dir, '*.swc')):
              swc_df = read_swc_pandas(swc_file)
              file_name = os.path.basename(swc_file)
              print(f'PROCESSING FILE: {file_name}')
              try:
                 G = create_graph_nx(swc_df)
              except Exception as e:
                print(f"Error creating graph for file {swc_file}: {e}")
                continue

              threshold = get_threshold(file_name, file_metrics)
           
              swc_in = swc_file
              file_name = os.path.basename(swc_file)
              #print(f'FILE: {file_name}')
              swc_out = os.path.join(SWC_dir, f'{os.path.splitext(os.path.basename(swc_file))[0]}.connected.swc')

              num_long_connections = 0
              if check_long_connections:           
                  for file_data in file_metrics:
                     if file_data['file_name'] == file_name:
                        num_long_connections = file_data['long_connections']
                        break
       
              start_time = time.time()

              # Identify the main tree and isolated trees
              main_tree, isolated_trees = identify_trees(G)
           
              if isolated_trees: 
                 num_isolated_trees = len(isolated_trees)
                 #print(f"*******Number of isolated trees: {num_isolated_trees}")
           
              # Output the results
              #print(f"Main Tree (Soma Node Included): {main_tree}")
        
              # Find the soma node (node_type=1, parent=-1)
              soma_node = find_soma(G)
              if soma_node is None:
                  print("Soma node not found in the graph.")
                  continue

              # Print the node type of each node in the main tree
              #print("Main Tree Node Types:")
              #for node in main_tree:
              #    node_type = G.nodes[node]['node_type']
              #    print(f"Node ID: {node}, Node Type: {node_type}")
        
              # Get and print the node_type for the main tree
              #main_tree_type = get_dominant_node_type(G, main_tree)
              #print(f"Main Tree Type: {main_tree_type} with node IDs: {list(main_tree)}")
        
              #for i, tree in enumerate(isolated_trees, 1):
              #    # Get and print the node_type for each isolated tree
              #    isolated_tree_type = get_dominant_node_type(G, tree)
              #    print(f"Isolated Tree {i} Type: {isolated_tree_type} with node IDs: {list(tree)}\n")
            
              # Print the parent and leaves of each isolated tree
              for i, tree in enumerate(isolated_trees, 1):
                  # Find the parent node for the isolated tree (this would be the first node or the root node of the tree)
                  parent_node = None
                  for node in tree:
                      if G.nodes[node]['parent'] == -1:
                          parent_node = node
                          break
            
                  min_distance = float('inf')
                  closest_node = None
                  for node in tree:  # Loop through all nodes in the isolated tree
                      distance = euclidean_distance(G.nodes[soma_node], G.nodes[node])
                      if distance is not None and distance < min_distance:
                          min_distance = distance
                          closest_node = node

                  # Calculate the Euclidean distance between soma_node and closest_node
                  soma_pos = np.array(G.nodes[soma_node]['pos'])
                  closest_node_pos = np.array(G.nodes[closest_node]['pos'])
                  distance = euclidean(soma_pos, closest_node_pos)
    
                  # Check if the distance is within the threshold
                  if distance <= (threshold):
                      #print(f"Isolated Tree {i}: Closest Node to Soma: Node ID {closest_node} with Position {closest_node_pos} and distance {distance:.2f}")                   
                      # Connect the closest node to the soma by updating its parent
                      connect_single_subtree(G, closest_node)
                                            
              # Write the updated graph to a new SWC file with '_connected' appended to the original file name
              write_swc(G, swc_in)
              #print(f"Updated SWC file written to: {swc_in}")

              #print(f"****threshold: {threshold}")
              threshold_factor = 1.5
              max_iterations = 10
              iteration = 0

              # Loop until isolated trees are no longer found or we reach the maximum number of iterations
              while iteration < max_iterations:
                 #cmd = f'sh Vaa3D-x.sh -x {lib} -f connect_neuron_SWC -i {swc_in} -o {swc_out} -p 0 100'
                 #os.system(f'sh Vaa3D-x.sh -x {lib} -f connect_neuron_SWC -i {swc_in} -o {swc_out} -p 0 100 > /dev/null 2>&1')
                 os.system(f'sh Vaa3D-x.sh -x {lib} -f connect_neuron_SWC -i {swc_in} -o {swc_out} -p 0 {threshold} > /dev/null 2>&1')
           
                 # Reload the SWC file after processing
                 G = create_graph_nx(read_swc_pandas(swc_out))
           
                 # Re-evaluate and Identify the main tree and isolated trees
                 main_tree, isolated_trees = identify_trees(G)
              
                 # Check if isolated trees exist
                 if isolated_trees:
                     #print(f"Isolated trees found with threshold {threshold}. Increasing threshold...")
                     #print("Connecting any disconnected segments...")
                     threshold *= threshold_factor  # Increase the threshold by the specified factor
                     iteration += 1  # Increment the iteration counter
                 else:
                     #print("No disconnected segments found. Process complete.")
                     break
           
              if check_long_connections and num_long_connections > 0:
                 if num_long_connections == 1:
                    print(f"From a total of {num_long_connections} long connection, removed {num_long_connections} long connection")
                 else:
                    print(f"From a total of {num_long_connections} long connections, removed {num_long_connections} long connections")
    
              end_time = time.time()
              elapsed_time = end_time - start_time

              set_elapse_time(file_name, elapsed_time, file_metrics)

              if elapsed_time < 1:
                # Convert time to milliseconds
                elapsed_time_ms = elapsed_time * 1000
                print(f'PROCESSED IN {elapsed_time_ms:.2f} milliseconds\n')
              else:
                # Display time in seconds
                print(f'PROCESSED IN {elapsed_time:.2f} seconds\n')
    
           # CSV file path and headers
           timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
           csv_file_path = '/nmo_swc/SWC/' + f'{archive_folder_name}-process-log-{timestamp}.csv'
           csv_headers = ['File name', 'Mean distance', 'StDev', 'Num long connections', 'Num long connections removed', 'Num isolated node removed ', 'Processing time']

           # Open the CSV file for writing
           with open(csv_file_path, 'w') as csvfile:
               # Write the headers to the CSV file
               csvfile.write(','.join(csv_headers) + '\n')
         
               # Format mean_distance and std_distance to 5 digits after the decimal point
               mean_distance = f"{float(file_data['mean_distance']):.5f}" if 'mean_distance' in file_data and file_data['mean_distance'] != 'N/A' else 'N/A'
               std_distance  = f"{float(file_data['std_distance']):.5f}" if 'std_distance' in file_data and file_data['std_distance'] != 'N/A' else 'N/A'
        
               # Write each file's data to the CSV
               for file_data in file_metrics:                
                   row = [
                       file_data['file_name'],                                     # File name
                       f"{file_data.get('mean_distance', 'N/A'):.5f}" if isinstance(file_data.get('mean_distance', None), (int, float)) else 'N/A',  # Mean distance
                       f"{file_data.get('std_distance', 'N/A'):.5f}" if isinstance(file_data.get('std_distance', None), (int, float)) else 'N/A', # Standard deviation
                       str(file_data.get('long_connections', 'N/A')),              # Number of long connections
                       str(file_data.get('num_long_connections_removed', 'N/A')),  # Number of long connections removed
                       str(file_data.get('num_isolated_nodes', 'N/A')),            # Number of isolated node removed 
                       str(file_data.get('elapsed_time', 'N/A'))                   # Processing time
                   ]
        
                   # Write the row to the CSV file
                   csvfile.write(','.join(row) + '\n')

           # Sort neurons after connection of Disjoint subtrees
           os.system('python3 sort.py')
        
           # Move result of connected subtrees to output
           os.system(f'mv /nmo_swc/SWC/*.* /nmo_swc/output/')

           # Write not processed to control file
           #source_version_dir = f'/nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version'
           source_version_dir = f'/nmo_swc/Source-Version'
           CNG_dir = '/nmo_swc/out_Final/CNG_Version/'
           source_files = get_files_list(source_version_dir)
           cng_files = get_files_list(CNG_dir)

           source_files = [os.path.basename(f) for f in source_files]
           cng_files = [os.path.basename(f) for f in cng_files]

           # Get file names without extensions
           source_files_without_ext = []
           for file in source_files:
               if file.lower().endswith('.swc'):
                   try:
                       source_files_without_ext.append(get_filename_without_extension(file))
                   except Exception as e:
                       print(f"Error getting source file: {file} - {e}")

           cng_files_without_ext = []
           for file in cng_files:
               # accept either .CNG.swc or plain .swc if your out folder varies
               if file.lower().endswith('.cng.swc'):
                   try:
                       cng_files_without_ext.append(get_filename_without_extension(file))
                   except Exception as e:
                       print(f"Error with processing CNG file: {file} - {e}")

           # Find missing files (NO PATHS POSSIBLE HERE)
           try:
               missing_files = sorted(set(source_files_without_ext) - set(cng_files_without_ext))
           except Exception as e:
               print(f"Error finding missing files: {e}")
               missing_files = []

           # Collect invalid files
           invalid_dir = '/nmo_swc/invalid-swc'
           if os.path.exists(invalid_dir) and os.path.isdir(invalid_dir):
               invalid_files = sorted(
                   os.path.basename(p)
                   for p in glob.glob(os.path.join(invalid_dir, "*"))
               )
           else:
               invalid_files = []

           # Write result to control file
           try:
               control_file = f'/nmo_swc/output/control.txt'
               os.makedirs(os.path.dirname(control_file), exist_ok=True)
               if os.path.exists(control_file):
                   os.remove(control_file)

               with open(control_file, "w") as f:
                   f.write(f"Total files in Source-version folder: {len(source_files)}\n")
                   f.write(f"Total processed files in CNG-Version folder: {len(cng_files_without_ext)}\n\n")

                   elapsed = time.time() - overall_start_time
                   h = int(elapsed // 3600)
                   m = int((elapsed % 3600) // 60)
                   s = int(elapsed % 60)
                   f.write(f"Total processing time: {h:02d}:{m:02d}:{s:02d} ({int(elapsed)} seconds)\n\n")

                   f.write("Files in Source-version not processed:\n")
                   for name in missing_files:
                       f.write(f"{name}.swc\n")

                   f.write("\nInvalid files moved to invalid-swc:\n")
                   if invalid_files:
                       for name in invalid_files:
                           f.write(f"{name}\n")
                   else:
                       f.write("None\n")

           except Exception as e:
               print(f"Error writing to control file: {e}")


           # Move all log files to {archive_folder_name}_Final
           os.system(f'cp /nmo_swc/Normalized/Possible-issues/Log.txt /nmo_swc/output/Log1.txt')
           os.system(f'cp /nmo_swc/out_Final/Remaining_issues/Log.txt /nmo_swc/output/Log2.txt')

           # Clear swc_trees_dir '/nmo_swc/swc_trees_dir'
           swc_trees_dir = '/nmo_swc/swc_trees_dir'
           clear_folder(swc_trees_dir)
    
           # Clear long_connect_dir '/nmo_swc/long_connect'
           #print(f'Clearing {long_connect_dir}\n')
           long_connect_dir = '/nmo_swc/long_connect'
           clear_folder(long_connect_dir)

           print(f"\nProcessing status successfully written to csv log file")

           print('\n**DISJOINT SUBTREES CONNECTION COMPLETED**')

           return jsonify({
            "status": "success",
            "message": "Disjoint Subtrees Connection Completed Successfully"
           }), 200

    except Exception as e:
        print("DISJOINT SUBTREES CONNECTION FAILED:", str(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        end_process()


# Function to get the threshold for a given file name
def get_threshold(file_name, file_metrics):
    # Search for the file in the list of metrics
    for file_data in file_metrics:
        file_name1 = file_data['file_name']
        if file_data['file_name'] == file_name:
            return file_data['threshold']
    # Return None if the file is not found
    return None

# Function to set the Elapse Time 
def set_elapse_time(file_name, elapsed_time, file_metrics):
    # Format the elapsed time based on its value
    if elapsed_time < 1:  # Less than 1 second
        formatted_time = "{:.2f} milliseconds".format(elapsed_time * 1000)  # Convert to milliseconds
    else:
        formatted_time = "{:.2f} seconds".format(elapsed_time)  # Keep in seconds

    # Search for the file in the list of metrics
    for file_data in file_metrics:
        if file_data['file_name'] == file_name:
            file_data['elapsed_time'] = formatted_time
            break

# Function to set Num connections removed 
def set_num_long_connections_removed(file_name, num_connections, file_metrics):
    for file_data in file_metrics:
       if file_data['file_name'] == file_name:
          file_data['num_long_connections_removed'] = num_connections
          break

# Function to read SWC file using pandas and create a DataFrame
def read_swc_pandas(filename):
   try:
      swc_df = pd.read_csv(filename, sep=r'\s+', comment='#', header=None,
                           names=['node_id', 'node_type', 'x', 'y', 'z', 'radius', 'parent'])

      swc_df['node_id'] = swc_df['node_id'].astype(int)
      swc_df['node_type'] = swc_df['node_type'].astype(int)
      swc_df['parent'] = swc_df['parent'].astype(int)

      return swc_df
   except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

# Function to create a NetworkX graph from SWC data
def create_graph_nx(swc_df):
    G = nx.Graph()

    for idx, row in swc_df.iterrows():
        node_id = int(row['node_id'])
        parent_id = int(row['parent'])
        pos = (row['x'], row['y'], row['z'])
        node_type = int(row['node_type'])
        radius = row['radius']

        # Add the node with all attributes
        G.add_node(node_id, pos=pos, node_type=node_type, radius=radius, parent=parent_id, **{k: v for k, v in row.items() if k not in ['node_id', 'node_type', 'x', 'y', 'z', 'radius', 'parent']})

        # Handle soma node separately
        if parent_id == -1:
            # Add a self-loop or a dummy edge
            G.add_edge(node_id, node_id)
        else:
            G.add_edge(parent_id, node_id)

    return G

# Function to calculate Euclidean distance between two nodes
def euclidean_distance(node1, node2):
    try:
       pos1 = np.array(node1['pos'])
       pos2 = np.array(node2['pos'])
       distance = euclidean(pos1, pos2)
       return distance
    except KeyError as e:
        print(f"Missing position key in node: {e}")
        return None
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return None

# Function to write the SWC file from a NetworkX graph
def write_swc(G, filename):
    with open(filename, 'w') as file:
        for node in G.nodes(data=True):
            node_id = int(node[0])
            node_data = node[1]
            parent_id = int(node_data['parent'])
            x, y, z = node_data['pos']
            radius = node_data['radius']
            node_type = int(node_data['node_type'])

            file.write(f"{node_id} {node_type} {x} {y} {z} {radius} {parent_id}\n")

def empty_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        else:
            print(f"The folder {folder_path} does not exist.")
    except Exception as e:
        print(f"Error emptying folder: {str(e)}")

def clear_folder(folder_path):
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove all files and subdirectories within the folder
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    except Exception as e:
        # Log an error message if an exception occurs
        print(f'Error clearing folder: {str(e)}')

def copy_file(source_path, destination_path):
    with open(source_path, 'rb') as source_file, open(destination_path, 'wb') as dest_file:
        dest_file.write(source_file.read())

def SWC_LONG_CONNECTIONS_STD():

    response = Response(status=200)
    
    # Check if the upload directory is not empty
    upload_dir = '/nmo_swc/Source-Version'
    swc_files = glob.glob(os.path.join(upload_dir, '*.swc'))
    if not swc_files:
        print()
        print('Please Upload .swc files to proceed...\n')
        return {"error_code": 404, "message": "No .swc files found in the upload directory."}

    # convert tags 5/6/7 -> 3 BEFORE Normalize.jar (store restore map per file)
    restore_maps = {}  # {basename.swc: {node_id: original_type}}
    for fp in swc_files:
        rm = _convert_tags_to_3_by_node_id_inplace(fp)
        if rm:
            print(f"[TAGFIX] {os.path.basename(fp)} contains node_type in {sorted(TAG_SET)} -> temporarily set to {TEMP_TAG}", flush=True)
            restore_maps[os.path.basename(fp)] = rm

    # Run Normalize.jar with swc stage in Source-Version folder
    main_dir = '/nmo_swc'
    os.chdir(main_dir)

    if os.path.exists("Normalized"):
        dir_name = 'Normalized'
        subprocess.run(['sudo', 'chmod', '-R', '777', dir_name], check=True)
        shutil.rmtree("Normalized")    
    os.makedirs("Normalized")
    os.chmod("Normalized", 0o777)
    
    if os.path.exists("Standardizationlog"):
        dir_name ='Standardizationlog'
        subprocess.run(['sudo', 'chmod', '-R', '777', dir_name], check=True)
        shutil.rmtree("Standardizationlog")
    os.makedirs("Standardizationlog")
    os.chmod("Standardizationlog", 0o777)

    if os.path.exists("Temp_LMeasure"):
        dir_name = 'Temp_LMeasure'
        subprocess.run(['sudo', 'chmod', '-R', '777', dir_name], check=True)
        shutil.rmtree("Temp_LMeasure")
    os.makedirs("Temp_LMeasure")
    os.chmod("Temp_LMeasure", 0o777)

    print (' ')
    os.chdir(main_dir)

    norm_dir = '/nmo_swc/Normalized'

    os.system('java -jar Normalize.jar')
    os.system('java -jar Check_norm.jar')

    # restore original tags AFTER Check_norm.jar (in Normalized outputs) ---
    if restore_maps and os.path.isdir(norm_dir):
        for norm_fp in glob.glob(os.path.join(norm_dir, "*.swc")):
            base = os.path.basename(norm_fp)
            if base in restore_maps:
                #print('\n***Restoring Tag...')
                _restore_tags_by_node_id_inplace(norm_fp, restore_maps[base]) 

    wkdir = os.getcwd()

    swc_files = glob.glob('/nmo_swc/Normalized/*.swc')
    if swc_files:
       # Remove all files in duplicate_remover
       dup_remover_dir = 'duplicate_remover'
       subprocess.run(['sudo', 'chmod', '-R', '777', dup_remover_dir], check=True)
       os.system('rm -rf /nmo_swc/duplicate_remover/*.swc')

       # Copy SWC files from Normalized to duplicate_remover
       os.system('cp /nmo_swc/Normalized/*.swc /nmo_swc/duplicate_remover/')

       # Copy Log.txt from Normalized/Possible-issues to duplicate_remover
       source_dir = '/nmo_swc/Normalized/Possible-issues'
       chmod_command = f'sudo chmod -R 777 {source_dir}'
       os.system(chmod_command)
       os.system('cp /nmo_swc/Normalized/Possible-issues/Log.txt /nmo_swc/duplicate_remover/')
       
       # Change files permission
       #os.system('sudo chmod -R 755 /nmo_swc/duplicate_remover')
       
       # Set working directory to /nmo_swc/duplicate_remover
       working_dir = '/nmo_swc/duplicate_remover'
       os.chdir(working_dir)
       
       # Run duplicate removal for 2.6 code
       print('\n***Running Duplicate Removal for 2.6 code...')
       search_string = '2.6=>'
       log_file_path = '/nmo_swc/duplicate_remover/Log.txt'

       if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
          result = subprocess.run(['grep', search_string, log_file_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
                           
          if result.returncode == 0:
             print(f'Code "{search_string}" was found in the log file.')
             print(result.stdout.strip())
             os.system('sh /nmo_swc/duplicate_remover/sduplicate.sh')
             # Copy *.swc from duplicate_remover to Normalized folder to overwrite faulty *.swc
             os.system('cp -f /nmo_swc/duplicate_remover/*.swc /nmo_swc/Normalized')
          else:
             print(f'No 2.6 code found in log file.')
       else:
             print(f'No 2.6 code found in log file.')

       # Set working directory to /nmo_swc
       working_dir = '/nmo_swc'
       os.chdir(working_dir)

       dir_name = 'nmo_user'
       if not os.path.exists(dir_name):
         os.makedirs(dir_name)
       chmod_command = f'sudo chmod -R 777 {dir_name}'
       os.system(chmod_command)
 
       # Remove all files in nmo_user
       os.system('rm -rf /nmo_swc/nmo_user/*')

       dir_user_final = '/nmo_swc/nmo_user_Final'
       if not os.path.exists(dir_name):
         os.makedirs(dir_user_final)
       chmod_command = f'sudo chmod -R 777 {dir_user_final}'
       os.system(chmod_command)

       # Remove all files in nmo_user_Final
       os.system('rm -rf /nmo_swc/nmo_user_Final/*')

       # Copy *.swc files from Normalized to Folder user
       os.system('cp /nmo_swc/Normalized/*.swc /nmo_swc/nmo_user')

       remaining_issues_dir = '/nmo_swc/nmo_user_Final/Remaining_issues'
       # Create or clear the directory if it exists       
       if os.path.exists(remaining_issues_dir):
          subprocess.run(['sudo', 'chmod', '-R', '777', remaining_issues_dir], check=True)
          shutil.rmtree(remaining_issues_dir)
       os.makedirs(remaining_issues_dir)
       chmod_command = f'sudo chmod -R 777 {remaining_issues_dir}'
       os.system(chmod_command)

       # Convert tags 5/6/7 -> 3 in the SAME folder passed to Finalize.jar ---
       restore_maps_finalize = {}  # {filename.swc: {node_id: original_type}}
       finalize_in_dir = f'/nmo_swc/{dir_name}'

       for fp in glob.glob(os.path.join(finalize_in_dir, "*.swc")):
           rm = _convert_tags_to_3_by_node_id_inplace(fp)
           if rm:
               #print(f"{os.path.basename(fp)} contains node_type in {sorted(TAG_SET)} -> temporarily set to {TEMP_TAG}", flush=True)
               restore_maps_finalize[os.path.basename(fp)] = rm

       # Run Finalize
       os.system(f'java -jar Finalize.jar {dir_name}')     

       # Run Check to generate log file
       os.system('java -jar Check.jar')

       # Restore original tags AFTER Check.jar (in nmo_user_Final outputs)
       cng_out_dir = '/nmo_swc/nmo_user_Final/CNG_Version'

       if restore_maps_finalize and os.path.isdir(cng_out_dir):
           for fp in glob.glob(os.path.join(cng_out_dir, "*.swc")):
               base = os.path.basename(fp)
               if base in restore_maps_finalize:
                       _restore_tags_by_node_id_inplace(fp, restore_maps_finalize[base])

       print('\n***Running Side Branch Deletion for 2.7 code...')

       # Copy Side_Branch_Del.py from /nmo_swc to nmo_user_Final       
       os.system('cp /nmo_swc/Side_Branch_Del.py /nmo_swc/nmo_user_Final')
       file_path = '/nmo_swc/nmo_user_Final/Side_Branch_Del.py'
       chmod_command = ['sudo', 'chmod', '777', file_path]       
       subprocess.run(chmod_command, check=True)

       remaining_issues_dir = '/nmo_swc/nmo_user_Final/Remaining_issues'
       log_file = os.path.join(remaining_issues_dir, 'Log.txt')
       destination_dir = '/nmo_swc/nmo_user_Final'
       destination_log_file = os.path.join(destination_dir, 'Log.txt')

       # Ensure the directory and its contents are readable, writable, and executable by all
       os.system(f'chmod -R 777 {remaining_issues_dir}')

       # Copy the Log.txt file from remaining_issues to the destination directory
       os.system(f'cp -f {log_file} {destination_dir}')

       # Make sure the copied Log.txt file in the destination directory is executable
       os.system(f'sudo chmod 777 {destination_log_file}')

       search_string = '2.7=>'
       log_file_path = '/nmo_swc/nmo_user_Final/Log.txt'
       if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:          
          result = subprocess.run(['grep', search_string, log_file_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)                    

          # Set working directory to /nmo_swc/nmo_user_Final
          working_dir = '/nmo_swc/nmo_user_Final'
          os.chdir(working_dir)

          # Run Side Branch Deletion
          if result.returncode == 0:
             print(f'Code "{search_string}" was found in the log file.')
             print(result.stdout.strip())
             os.system('python3 Side_Branch_Del.py /nmo_swc/nmo_user_Final/Remaining_issues')
          else:
             print(f'No 2.7 code found in log file.')
       else:
          print(f'No 2.7 code found in log file.')
              
       # Set working directory to /nmo_swc
       working_dir = '/nmo_swc'
       os.chdir(working_dir)

       # Create out directory
       out_dir_name = 'out'
       if os.path.exists(out_dir_name):
          subprocess.run(['sudo', 'chmod', '-R', '777', out_dir_name], check=True)
          shutil.rmtree(out_dir_name)
       os.makedirs(out_dir_name)
       os.chmod(out_dir_name, 0o777)
       
       # Remove out_Final directory
       if os.path.exists('/nmo_swc/out_Final/'):
          out_Final_dir = '/nmo_swc/out_Final/'
          subprocess.run(['sudo', 'chmod', '-R', '777', out_Final_dir], check=True)
          shutil.rmtree(out_Final_dir)

       # Copy *.swc from /nmo_swc/nmo_user_Final/CNG to out
       os.system('cp -u /nmo_swc/nmo_user_Final/CNG_Version/*.swc /nmo_swc/out')

       # Copy *.swc from /nmo_swc/nmo_user_Final/out to out to overwrite older *.swc
       out_dir_path = '/nmo_swc/nmo_user_Final/out'
       if os.path.exists(out_dir_path) and os.path.isdir(out_dir_path):
          #print('out directory exist')
          os.system('cp -f /nmo_swc/nmo_user_Final/out/*.swc /nmo_swc/out')

       print('\n***Running radius fix for 4.1 code...')
       # Copy *.swc from /nmo_swc/out to output_radius
       output_radius_dir = '/nmo_swc/output_radius'
       cng_version_dir = os.path.join(output_radius_dir, 'CNG_Version') 
       remaining_issues_dir = os.path.join(output_radius_dir, 'Remaining_issues')
       
       # Clean out_radius directory if exist
       if os.path.exists(output_radius_dir):
          subprocess.run(['sudo', 'chmod', '-R', '777', output_radius_dir], check=True)
          shutil.rmtree(output_radius_dir)
       os.makedirs(output_radius_dir)
       
       if not os.path.exists(cng_version_dir):
          os.makedirs(cng_version_dir)

       if not os.path.exists(remaining_issues_dir):
          os.makedirs(remaining_issues_dir)
       
       os.system(f'cp -f /nmo_swc/out/*.swc {cng_version_dir}')
       os.system(f'cp -f /nmo_swc/nmo_user_Final/Remaining_issues/*.* {remaining_issues_dir}')
       os.system(f'cp -f /nmo_swc/nmo_user_Final/Log.txt {output_radius_dir}')

       os.system(f'sudo chmod -R 777 {cng_version_dir}')
       os.system(f'sudo chmod -R 777 {remaining_issues_dir}')
       os.system(f'sudo chmod -R 777 {output_radius_dir}')

       # Copy fix_radius.py from /nmo_swc to nmo_user_Final
       #working_dir = '/nmo_swc/nmo_user_Final'
       os.system('cp /nmo_swc/fix_radius.py /nmo_swc/output_radius/')
       os.system(f'sudo chmod 777 /nmo_swc/output_radius/fix_radius.py')
       
       search_string = '4.1=>'
       log_file_path = '/nmo_swc/output_radius/Log.txt'
       
       grep_command = f'grep "{search_string}" {log_file_path}'
       #exit_code = os.system(grep_command)
       exit_code = subprocess.call(grep_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

       # Run fix radius
       if exit_code == 0:
          print(f'Code string "{search_string}" was found in the log file.')

          # Set working directory to output_radius_dir
          os.chdir(output_radius_dir)          
          print("Current working directory:", os.getcwd())          

          os.system('python3 fix_radius.py /nmo_swc/output_radius/Remaining_issues')

          os.system(f'sudo chmod -R 777 /nmo_swc/output_radius/out_radius')

          # Copy swc.* from /nmo_swc/output_radius/out_radius to out to overwrite older *.swc
          out_radius_dir_path = '/nmo_swc/output_radius/out_radius'
          if os.path.exists(out_radius_dir_path) and os.path.isdir(out_radius_dir_path):
             #print('out directory exist')
             os.system(f'cp -f /nmo_swc/output_radius/out_radius/*.swc /nmo_swc/out')
             os.system(f'sudo chmod -R 777 /nmo_swc/out')
          print ('Completed')
       else:
          print(f'No 4.1 code found in log file.')

       # Convert tags 5/6/7 -> 3 BEFORE Finalize.jar (out)
       restore_maps_finalize_out = {}  # {filename.swc: {node_id: original_type}}

       finalize_out_in_dir = '/nmo_swc/out'
       for fp in glob.glob(os.path.join(finalize_out_in_dir, "*.swc")):
           rm = _convert_tags_to_3_by_node_id_inplace(fp)
           if rm:
               #print(f"[TAGFIX-FINALIZE-OUT] {os.path.basename(fp)} contains node_type in {sorted(TAG_SET)} -> temporarily set to {TEMP_TAG}", flush=True)
               restore_maps_finalize_out[os.path.basename(fp)] = rm

       # Run Finalize
       working_dir = '/nmo_swc'
       os.chdir(working_dir)
       out_dir_name = 'out'
       os.system(f'java -jar Finalize.jar {out_dir_name}')

       # Run Check again to generate log file
       os.system('java -jar Check.jar')

       # Restore original tags AFTER Check.jar (Finalize out output) ---       
       finalize_out_output_dir = '/nmo_swc/out_Final/CNG_Version'

       if restore_maps_finalize_out and os.path.isdir(finalize_out_output_dir):
           for fp in glob.glob(os.path.join(finalize_out_output_dir, "*.swc")):
               base = os.path.basename(fp)
               if base in restore_maps_finalize_out:
                   _restore_tags_by_node_id_inplace(fp, restore_maps_finalize_out[base])

       return {"status": "success"}
       
      
@app.route('/nmo/CorrectTag', methods=["GET", "POST"])
def CorrectTag():

    if not start_process():
        return jsonify({"error": "Another process is already running"}), 429

    start_time = time.time()

    try:
       # clear log content if exists
       file_path = '/nmo_swc/log/app.log'
       if os.path.exists(file_path):
          # Open the file in write mode, which clears its contents
          with open(file_path, 'w') as file:
             pass

       # Check if the upload directory is not empty
       upload_dir = '/nmo_swc/Source-Version'
       swc_files = glob.glob(os.path.join(upload_dir, '*.swc'))
       if not swc_files:
           print()
           print('Please Upload .swc files to proceed...\n')
           return "No .swc files found in the upload directory."

       print('\n***PROCESSING CORRECTION OF MISLABELED PYRAMIDALS...\n')
       
       main_dir = '/nmo_swc'
       in_dir_name = 'pyramidals_incorrect_tag_swc'
       out_dir_name = 'pyramidals_corrected_tag_swc'
       out_png_dir_name = 'pyramidals_corrected_tag_png'

       os.chdir(main_dir)
       if os.path.exists(in_dir_name):
           shutil.rmtree(in_dir_name)
       os.makedirs(in_dir_name, exist_ok=True)
       os.chmod(in_dir_name, 0o777)

       if os.path.exists(out_dir_name):
           shutil.rmtree(out_dir_name)
       os.makedirs(out_dir_name, exist_ok=True)
       os.chmod(out_dir_name, 0o777)

       if os.path.exists(out_png_dir_name):
           shutil.rmtree(out_png_dir_name)
       os.makedirs(out_png_dir_name, exist_ok=True)
       os.chmod(out_png_dir_name, 0o777)

       os.system(f'mv {upload_dir}/*.swc {main_dir}/{in_dir_name}/')

       files = [os.path.join(in_dir_name, f) for f in os.listdir(in_dir_name) if f.endswith('.swc')]

       # Process files sequentially
       processed_count = 0
       for idx, file_path in enumerate(files):
          swc_filename = os.path.basename(file_path)
          print(f"\n({idx+1}/{len(files)}) PROCESSING FILE: {swc_filename}")

          graph = read_swc_file(file_path)
          
          t0 = time.time()
          
          summary = assign_apical_basal_rule(graph, radial_step=10.0, debug=False)
          print(f"[main] Summary: {summary}")

          print(f"processing time: {time.time() - t0:.2f}s\n")

          output_path = os.path.join(out_dir_name, swc_filename)
          save_swc_file(graph, output_path)
          print(f"Corrected file saved at {output_path}")
          print('')

          processed_count += 1
    
       print('\n***PNG Automation In Progress...')
       PNG_dir = '/nmo_swc/PNG_Automation_SN/PNG_Automation_SN'
       os.chdir(PNG_dir)
       os.system(f'rm -rf {PNG_dir}/PNG/*.*')
       os.system(f'rm -rf {PNG_dir}/SWC/*.*')
       os.system(f'cp /nmo_swc/{out_dir_name}/*.swc {PNG_dir}/SWC/')
       os.system('sh /nmo_swc/PNG_Automation_SN/PNG_Automation_SN/PNG_generator.sh')

       wrk_dir = '/nmo_swc/'
       os.chdir(wrk_dir)

       '''
       if not is_tomcat_running():
          #print("Tomcat is not running. Starting Tomcat...")
          os.system("/opt/tomcat/bin/catalina.sh start")

          # Wait until Tomcat is ready
          #print("Waiting for Tomcat to complete startup...")
          max_wait = 60  # seconds
          start_time = time.time()
          while not is_tomcat_running():
             if time.time() - start_time > max_wait:
                #print("Tomcat is not running. Cannot generate PNG...")
                raise RuntimeError("Tomcat failed to start within 60 seconds")
             time.sleep(2)
          #print("Tomcat is up and running!")
       else:
          print("")
          #print("Tomcat is already running.")

       os.system('python3 process_swcs.py')
       '''
    
       png_files = glob.glob(f"{PNG_dir}/PNG/*.png")
       if png_files:
           os.system(f"mv {PNG_dir}/PNG/*.png /nmo_swc/{out_png_dir_name}/")

       # Remove all files in mounted output_Tag directory
       output_dir_name = '/nmo_swc/output_Tag'
       if os.listdir(output_dir_name):
         os.system(f'rm -rf {output_dir_name}/*')

       # Move original file and results to mounted output
       in_dir_name_mv = f'{output_dir_name}/pyramidals_incorrect_tag_swc'
       out_dir_name_mv = f'{output_dir_name}/pyramidals_corrected_tag_swc'
       out_png_dir_name_mv = f'{output_dir_name}/pyramidals_corrected_tag_png'

       if os.path.exists(in_dir_name_mv):
           shutil.rmtree(in_dir_name_mv)
       os.makedirs(in_dir_name_mv, exist_ok=True)
       os.chmod(in_dir_name_mv, 0o777)

       if os.path.exists(out_dir_name_mv):
           shutil.rmtree(out_dir_name_mv)
       os.makedirs(out_dir_name_mv, exist_ok=True)
       os.chmod(out_dir_name_mv, 0o777)

       if os.path.exists(out_png_dir_name_mv):
           shutil.rmtree(out_png_dir_name_mv)
       os.makedirs(out_png_dir_name_mv, exist_ok=True)
       os.chmod(out_png_dir_name_mv, 0o777)

       os.system(f'mv /nmo_swc/{in_dir_name}/*.* {in_dir_name_mv}/')
       os.system(f'mv /nmo_swc/{out_dir_name}/*.* {out_dir_name_mv}/')
       src = f"/nmo_swc/{out_png_dir_name}/*.*"
       if glob.glob(src):
           os.system(f"mv {src} {out_png_dir_name_mv}/")

       elapsed = time.time() - start_time
       h = int(elapsed // 3600)
       m = int((elapsed % 3600) // 60)
       s = int(elapsed % 60)

       control_path = os.path.join(output_dir_name, "control.txt")
       with open(control_path, "w") as cf:
           cf.write("SWC Tag Correction Control File\n")
           cf.write("================================\n")
           cf.write("Total files processed: {}\n".format(processed_count))
           cf.write("Total processing time: {:02d}:{:02d}:{:02d} ({} seconds)\n".format(h, m, s, int(elapsed)))
       os.chmod(control_path, 0o777)
    
       print('\n**PNG GENERATION COMPLETED**')
       print('\n**SWC TAG CORRECTION COMPLETED**')
       
       return jsonify({
          "status": "success",
          "message": "SWC Tag Correction Completed Successfully"
       }), 200

    except Exception as e:
        print("SWC TAG CORRECTION FAILED:", str(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

    finally:
        end_process()


# SWC file reading function
def read_swc_file(filename: str) -> nx.DiGraph:
    """
    Read SWC into a networkx DiGraph with node attributes:
      node_type, x, y, z, radius, parent

    SAFEGUARDS:
      - Drops any row with NA/NaN/inf/-inf in required fields
      - Drops any row missing node_id / node_type / parent
      - Drops any row missing x/y/z
      - Coerces numeric safely
      - Fills missing/invalid radius with 1.0
      - Drops rows whose parent is not -1 and not present as a node_id in the file
    """
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["node_id", "node_type", "x", "y", "z", "radius", "parent"],
        engine="python",
        dtype=str,
    )

    # 1) Remove fully-empty lines
    df = df.dropna(how="all").copy()

    # 2) Coerce numeric (invalid -> NaN)
    cols = ["node_id", "node_type", "x", "y", "z", "radius", "parent"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Replace inf/-inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 4) Drop rows missing REQUIRED fields
    required = ["node_id", "node_type", "parent", "x", "y", "z"]
    df = df.dropna(subset=required).copy()

    if df.empty:
        raise ValueError(f"{os.path.basename(filename)}: SWC empty after removing invalid rows (NaN/inf/missing required fields).")

    # 5) Cast types safely (round ids)
    df["node_id"] = df["node_id"].round().astype(int)
    df["node_type"] = df["node_type"].round().astype(int)
    df["parent"] = df["parent"].round().astype(int)

    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["z"] = df["z"].astype(float)

    # radius: allow missing -> 1.0
    df["radius"] = pd.to_numeric(df["radius"], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["radius"] = df["radius"].fillna(1.0).astype(float)

    # 6) Optional: drop rows whose parent is not -1 and not present in node_id set
    #    (helps avoid creating "phantom" parent nodes later)
    node_ids = set(df["node_id"].tolist())
    bad_parent_mask = (df["parent"] != -1) & (~df["parent"].isin(node_ids))
    if bad_parent_mask.any():
        # drop those rows (or you could set parent=-1 instead; dropping is safer)
        df = df.loc[~bad_parent_mask].copy()

    if df.empty:
        raise ValueError(f"{os.path.basename(filename)}: SWC empty after removing rows with missing parents.")

    # 7) Build graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        nid = int(row["node_id"])
        parent = int(row["parent"])

        G.add_node(
            nid,
            node_type=int(row["node_type"]),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            radius=float(row["radius"]),
            parent=parent,
        )

    # Add edges only if parent exists (and parent != -1)
    for nid, data in G.nodes(data=True):
        parent = int(data.get("parent", -1))
        if parent != -1 and parent in G:
            G.add_edge(parent, nid)

    return G

# Function to save SWC file
def save_swc_file(graph, filename):
    print(f"Writing: {filename}  (nodes={graph.number_of_nodes()})")
    with open(filename, "w") as f:
        for i, node in enumerate(graph.nodes):
            data = graph.nodes[node]
            preds = list(graph.predecessors(node))
            parent = preds[0] if len(preds) > 0 else -1

            x = float(data["x"])
            y = float(data["y"])
            z = float(data["z"])

            f.write(
                f"{int(node)} {int(data['node_type'])} "
                f"{x:.2f} {y:.2f} {z:.2f} 1.0 {int(parent)}\n"
            )

            #if i > 0 and i % 200000 == 0:
            #    print(f"[save_swc_file] Wrote {i} nodes...")

    print("Done Writing File.")



def tree_centroid(graph, tree_nodes):
    xs = [graph.nodes[n]["x"] for n in tree_nodes]
    ys = [graph.nodes[n]["y"] for n in tree_nodes]
    zs = [graph.nodes[n]["z"] for n in tree_nodes]
    return np.array([float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))], dtype=float)


def euclid(a, b):
    return float(np.linalg.norm(a - b))


def has_only_allowed_node_types(filepath, allowed_types=ALLOWED_NODE_TYPES):
    """
    Returns True if every SWC row has node_type in allowed_types.
    Returns False if any row has an invalid node_type or the file is malformed.
    """
    try:
        swc_df = pd.read_csv(
            filepath,
            sep=r"\s+",
            comment="#",
            header=None,
            names=["node_id", "node_type", "x", "y", "z", "radius", "parent"],
            encoding="ISO-8859-1",
            engine="python",
        )

        # Ensure exactly 7 columns
        if swc_df.shape[1] != 7 or swc_df.empty:
            return False

        # Enforce data types (will raise if malformed)
        swc_df = swc_df.astype(
            {
                "node_id": int,
                "node_type": int,
                "x": float,
                "y": float,
                "z": float,
                "radius": float,
                "parent": int,
            }
        )

        # Check node_type validity
        invalid = swc_df.loc[~swc_df["node_type"].isin(allowed_types)]
        return invalid.empty

    except Exception:
        return False



# ------------------------------------------------------------
# Config / feature list
# ------------------------------------------------------------
ALIGNED_FEATURE_COLS = [
    "node_count",
    "tip_count",
    "bifurcations",
    "max_euclid",
    "mean_euclid",
    "max_path_length",
    "total_length",
    "sholl_sum",
    "elongation",
    "principal_axis_x",
    "principal_axis_y",
    "principal_axis_z",
]

# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def _safe_int(v, default=-1) -> int:
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default

def _safe_float(v, default=0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
        return float(default)
    except Exception:
        return float(default)

def _node_xyz(G: nx.DiGraph, n: int) -> np.ndarray:
    return np.array(
        [
            _safe_float(G.nodes[n].get("x", 0.0), 0.0),
            _safe_float(G.nodes[n].get("y", 0.0), 0.0),
            _safe_float(G.nodes[n].get("z", 0.0), 0.0),
        ],
        dtype=float,
    )

def _edge_len(G: nx.DiGraph, u: int, v: int) -> float:
    return float(np.linalg.norm(_node_xyz(G, v) - _node_xyz(G, u)))


def ensure_soma_node(graph):
    root_nodes = [n for n in graph.nodes if int(graph.nodes[n].get("parent", 999999)) == -1]
    if not root_nodes:
        #print("[ensure_soma_node] ERROR: No root node (parent==-1) found.")
        return None

    soma_node = sorted(root_nodes)[0]

    if int(graph.nodes[soma_node].get("node_type", -1)) != 1:
        #print(f"[ensure_soma_node] Forcing node_type=1 for root soma node {soma_node}.")
        graph.nodes[soma_node]["node_type"] = 1

    if len(root_nodes) > 1:
        print(
            f"[ensure_soma_node] WARN: Multiple roots found (parent==-1): {sorted(root_nodes)}. "
            f"Using soma={soma_node}."
        )

    return soma_node

# ----------------------------
# Debug: print first N nodes of each tree
# ----------------------------
def print_tree_first_nodes(graph, trees, tree_root_map, n_first=5):
    """
    Prints the first n_first node IDs for each tree (sorted order),
    plus (node_id, node_type, parent) for quick inspection.
    """
    for i, tree_nodes in enumerate(trees):
        first_ids = tree_nodes[:n_first]
        triples = []
        for nid in first_ids:
            nt = int(graph.nodes[nid].get("node_type", -1))
            parent = int(graph.nodes[nid].get("parent", -999))
            triples.append((int(nid), nt, parent))
        print(
            f"[tree-first] idx={i} root={tree_root_map[i]} nodes={len(tree_nodes)} "
            f"first_{n_first}_ids={first_ids} first_{n_first}_(id,type,parent)={triples}"
        )


def compute_tree_metrics(graph, tree_nodes, root_child):
    """
    Compute branch-aware metrics for ONE soma-child tree.

    Metrics:
      - total_cable: sum of Euclidean lengths of all edges in the tree
      - max_path:    maximum root_child -> any node path length (cable distance)
      - branch_pts:  count of branch points (out_degree >= 2) within this tree
      - tips:        count of terminal tips (out_degree == 0) within this tree
      - nodes:       number of nodes in the tree
      - trunk_len:   path length from root_child until first branch point (or tip)
    """
    T = tree_nodes
    Tset = set(T)

    children = {n: [] for n in T}
    for n in T:
        preds = list(graph.predecessors(n))
        p = preds[0] if preds else -1
        if p in Tset:
            children[p].append(n)

    # total cable length (sum of edges in the induced tree)
    total_cable = 0.0
    for p in T:
        for c in children.get(p, []):
            total_cable += _edge_len(graph, p, c)

    # root->node distances
    dist = {root_child: 0.0}
    stack = [root_child]
    while stack:
        u = stack.pop()
        for v in children.get(u, []):
            dist[v] = dist[u] + _edge_len(graph, u, v)
            stack.append(v)

    max_path = max(dist.values()) if dist else 0.0

    branch_pts = sum(1 for n in T if len(children.get(n, [])) >= 2)
    tips = sum(1 for n in T if len(children.get(n, [])) == 0)

    # trunk length: from root_child until first branch point or tip
    trunk_len = 0.0
    cur = root_child
    while True:
        ch = children.get(cur, [])
        if len(ch) != 1:
            break
        nxt = ch[0]
        trunk_len += _edge_len(graph, cur, nxt)
        cur = nxt

    return {
        "nodes": int(len(T)),
        "total_cable": float(total_cable),
        "max_path": float(max_path),
        "branch_pts": int(branch_pts),
        "tips": int(tips),
        "trunk_len": float(trunk_len),
    }


def select_apical_tree_branch_score(
    graph,
    soma_node,
    trees,
    eligible_idx,
    tree_root_map,
    debug=False,
):
    
    if len(eligible_idx) == 1:
        return eligible_idx[0]

    # ---- compute stats ----
    stats = {}
    for i in eligible_idx:
        root = tree_root_map[i]
        stats[i] = compute_tree_metrics(graph, trees[i], root)
        mp = max(stats[i]["max_path"], 1e-9)
        stats[i]["bushiness_ratio"] = float(stats[i]["total_cable"] / mp)
        stats[i]["trunk_ratio"] = float(stats[i]["trunk_len"] / mp)

    # ---- candidate filter ----
    candidates = apical_candidate_filter(stats, eligible_idx, tree_root_map, debug=debug)
    if not candidates:
        candidates = list(eligible_idx)

    # ----------------------------
    # TRUNK / REACH LOGIC
    # ----------------------------
    max_reach = max(stats[i]["max_path"] for i in candidates) if candidates else 0.0

    TRUNK_MIN_ABS = 20.0
    TRUNK_MIN_RATIO = 0.08
    REACH_FRAC_OF_MAX = 0.45
    reach_min = REACH_FRAC_OF_MAX * max_reach

    trunky = []
    for i in candidates:
        m = stats[i]
        if (m["trunk_len"] >= TRUNK_MIN_ABS) and (m["trunk_ratio"] >= TRUNK_MIN_RATIO) and (m["max_path"] >= reach_min):
            trunky.append(i)

    def reach_first_key(i):
        m = stats[i]
        return (m["max_path"], m["trunk_len"], m["nodes"])

    def trunk_first_key(i):
        m = stats[i]
        return (m["trunk_len"], m["max_path"], m["nodes"])

    # ----------------------------
    # trunk winner "too linear" suppression
    # ----------------------------
    LINEAR_MAX_BRANCH_PTS = 4
    LINEAR_MAX_TIPS = 5

    if trunky:
        trunk_winner = max(trunky, key=trunk_first_key)
        mw = stats[trunk_winner]
        looks_too_linear = (mw.get("branch_pts", 0) <= LINEAR_MAX_BRANCH_PTS) and (mw.get("tips", 0) <= LINEAR_MAX_TIPS)

        if looks_too_linear:
            apical = max(candidates, key=reach_first_key)
            decision_mode = "FALLBACK (reach-first; trunk suppressed: too linear)"
        else:
            apical = trunk_winner
            decision_mode = "TRUNK-OVERRIDE (trunk-first)"
    else:
        apical = max(candidates, key=reach_first_key)
        decision_mode = "FALLBACK (reach-first)"

    # -----------------------------------------------------------------
    # spindly reach-first winner vs substantial near-reach
    # -----------------------------------------------------------------
    SPINDLY_BRANCH_MAX = 4
    SPINDLY_TIPS_MAX = 5
    NEAR_REACH_FRAC = 0.85
    MIN_CABLE_MULT = 1.35
    MIN_NODE_MULT = 1.20
    MIN_BRANCH_DELTA = 6

    if decision_mode.startswith("FALLBACK"):
        mw = stats[apical]
        apical_is_spindly = (mw.get("branch_pts", 0) <= SPINDLY_BRANCH_MAX) and (mw.get("tips", 0) <= SPINDLY_TIPS_MAX)

        if apical_is_spindly:
            reach_cut = NEAR_REACH_FRAC * max(mw.get("max_path", 0.0), 1e-9)

            better = []
            for j in candidates:
                if j == apical:
                    continue
                mj = stats[j]
                if mj["max_path"] < reach_cut:
                    continue
                if mj["total_cable"] < (MIN_CABLE_MULT * mw["total_cable"]):
                    continue
                if mj["nodes"] < (MIN_NODE_MULT * mw["nodes"]):
                    continue
                if mj["branch_pts"] < (mw["branch_pts"] + MIN_BRANCH_DELTA):
                    continue
                better.append(j)

            if better:
                apical2 = max(better, key=lambda j: (stats[j]["total_cable"], stats[j]["nodes"], stats[j]["max_path"]))
                if debug:
                    print(
                        "\n[branch-apical] NOTE: reach-first picked a spindly tree; "
                        "switching to substantial near-reach candidate."
                    )
                apical = apical2
                decision_mode = "NEAR-REACH SUBSTANTIAL override (spindly reach-first fix)"

  
    if decision_mode.startswith("FALLBACK"):
        mw = stats[apical]
        winner_reach = max(mw.get("max_path", 0.0), 1e-9)
        winner_bushy = max(mw.get("bushiness_ratio", 0.0), 1e-9)
        winner_nodes = max(mw.get("nodes", 1), 1)
        winner_cable = max(mw.get("total_cable", 0.0), 1e-9)

        BUSHY_NEAR_REACH_FRAC = 0.70     # 339 vs 4 is ~0.73 so it qualifies
        BUSHY_RATIO_MULT = 1.30         # require clearly bushier
        BUSHY_MIN_NODE_MULT = 1.05      # roughly comparable size
        BUSHY_MIN_CABLE_MULT = 1.05     # roughly comparable size

        bushy_better = []
        for j in candidates:
            if j == apical:
                continue
            mj = stats[j]

            if mj["max_path"] < (BUSHY_NEAR_REACH_FRAC * winner_reach):
                continue
            if mj.get("bushiness_ratio", 0.0) < (BUSHY_RATIO_MULT * winner_bushy):
                continue
            if mj["nodes"] < (BUSHY_MIN_NODE_MULT * winner_nodes):
                continue
            if mj["total_cable"] < (BUSHY_MIN_CABLE_MULT * winner_cable):
                continue

            bushy_better.append(j)

        if bushy_better:
            # choose the bushiest among the qualifying near-reach candidates
            apical2 = max(
                bushy_better,
                key=lambda j: (
                    stats[j]["bushiness_ratio"],
                    stats[j]["total_cable"],
                    stats[j]["nodes"],
                    stats[j]["max_path"],
                ),
            )
            if debug:
                print(
                    "\n[branch-apical] NOTE: reach-first winner beaten by much bushier near-reach candidate; "
                    "switching to bushiness override."
                )
            apical = apical2
            decision_mode = "BUSHINESS override (near-reach, size-qualified)"

    # ---- debug prints ----
    if debug:
        print("\n[branch-apical] Decision mode:", decision_mode)
        for i in eligible_idx:
            m = stats[i]
            flags = []
            flags.append("CANDIDATE" if i in candidates else "EXCLUDED")
            if i in trunky:
                flags.append("TRUNKY")
            if i == apical:
                flags.append("<<APICAL")
            print(
                f"  idx={i} root={tree_root_map[i]} nodes={m['nodes']} "
                f"cable={m['total_cable']:.2f} max_path={m['max_path']:.2f} "
                f"branch_pts={m['branch_pts']} tips={m['tips']} trunk={m['trunk_len']:.2f} "
                f"trunk_ratio={m['trunk_ratio']:.3f} bushy={m['bushiness_ratio']:.2f} "
                f"{' '.join(flags)}"
            )
        print(f"[branch-apical] Decision: apical idx={apical} root={tree_root_map[apical]}\n")

    return apical




def build_soma_children_trees(graph: nx.DiGraph, soma_node: int) -> Tuple[List[List[int]], List[int]]:
    """
    Build list of node-lists (trees) for each child of soma. Return (trees, tree_root_map)
    Only include trees with >1 node (exclude degenerate singletons).
    """
    trees = []
    tree_root_map = []
    for child in graph.successors(soma_node):
        stack = [child]
        nodeset: Set[int] = set()
        while stack:
            cur = stack.pop()
            if cur == soma_node or cur in nodeset:
                continue
            nodeset.add(cur)
            stack.extend(list(graph.successors(cur)))
        if len(nodeset) > 0:
            trees.append(sorted(nodeset))
            tree_root_map.append(child)
    return trees, tree_root_map

# ------------------------------------------------------------
# Per-tree feature extraction including Sholl
# ------------------------------------------------------------
def compute_tree_topology_paths(G: nx.DiGraph, root: int, tree_nodes: Set[int]) -> Tuple[float, float, int]:
    """
    Compute:
      - max_path_length: longest path length from root to any tip (sum of edge lengths)
      - total_length: sum of all edge lengths inside tree
      - tip_count: number of tips (nodes with no children inside tree)
    """
    total_length = 0.0
    for u, v in G.edges():
        if u in tree_nodes and v in tree_nodes:
            total_length += _edge_len(G, u, v)
    tip_count = 0
    for n in tree_nodes:
        child_count = sum(1 for c in G.successors(n) if c in tree_nodes)
        if child_count == 0:
            tip_count += 1
    # compute longest root-to-tip path via DFS
    max_path = 0.0
    stack = [(root, 0.0)]
    while stack:
        node, acc = stack.pop()
        children = [c for c in G.successors(node) if c in tree_nodes]
        if not children:
            if acc > max_path:
                max_path = acc
        else:
            for c in children:
                stack.append((c, acc + _edge_len(G, node, c)))
    return float(max_path), float(total_length), int(tip_count)

def calculate_sholl_value(G: nx.DiGraph, tree_nodes: Set[int], soma_node: int, radial_step=10.0) -> float:
    """
    Compute a simple Sholl-like value: sum over rings of number of nodes in that ring.
    radial_step defines ring thickness. Returns sholl_sum (larger means more spread/branching across radii).
    """
    soma_coord = _node_xyz(G, soma_node)
    distances = [float(np.linalg.norm(_node_xyz(G, n) - soma_coord)) for n in tree_nodes]
    if not distances:
        return 0.0
    maxd = max(distances)
    if maxd <= 0.0:
        return 0.0
    bins = np.arange(0.0, maxd + radial_step, radial_step)
    counts = np.histogram(distances, bins=bins)[0]
    # return sum of counts weighted by ring index to emphasize nodes further away; or simply sum(counts)
    # Here use sum(counts) which is equal to node_count, but weighted sum gives more importance to spread.
    # weighted sum by ring index:
    ring_indices = np.arange(1, len(counts) + 1)
    weighted = float(np.sum(counts * ring_indices))
    return weighted

def compute_tree_features_aligned(
    G: nx.DiGraph,
    soma: int,
    root: int,
    tree_nodes: Set[int],
    radial_step: float = 10.0,
) -> Dict[str, Any]:
    """
    Compute aligned feature set for a tree (dictionary).
    """
    coords = np.array([_node_xyz(G, n) for n in tree_nodes], dtype=float) if tree_nodes else np.zeros((0, 3))
    soma_coord = _node_xyz(G, soma)
    dists = [float(np.linalg.norm(_node_xyz(G, n) - soma_coord)) for n in tree_nodes]
    max_euclid = float(max(dists)) if dists else 0.0
    mean_euclid = float(np.mean(dists)) if dists else 0.0
    max_path_length, total_length, tip_count = compute_tree_topology_paths(G, root, tree_nodes)
    sholl_sum = calculate_sholl_value(G, tree_nodes, soma, radial_step=radial_step)
    bifurcations = sum(1 for n in tree_nodes if sum(1 for c in G.successors(n) if c in tree_nodes) >= 2)
    node_count = int(len(tree_nodes))

    elongation = 0.0
    principal_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    try:
        if coords.shape[0] >= 2:
            centered = coords - coords.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            s0 = float(S[0]) if len(S) > 0 else 0.0
            s1 = float(S[1]) if len(S) > 1 else 1e-9
            elongation = float(s0 / (s1 + 1e-12))
            principal_axis = Vt[0] if Vt.shape[0] >= 1 else principal_axis
    except Exception:
        pass

    return {
        "node_count": node_count,
        "tip_count": int(tip_count),
        "bifurcations": int(bifurcations),
        "max_euclid": float(max_euclid),
        "mean_euclid": float(mean_euclid),
        "max_path_length": float(max_path_length),
        "total_length": float(total_length),
        "sholl_sum": float(sholl_sum),
        "elongation": float(elongation),
        "principal_axis_x": float(principal_axis[0]),
        "principal_axis_y": float(principal_axis[1]),
        "principal_axis_z": float(principal_axis[2]),
    }

# ------------------------------------------------------------
# Candidate gating and deterministic fallback
# ------------------------------------------------------------
def apical_candidate_filter(
    feats: Dict[int, Dict[str, Any]],
    eligible_idx: List[int],
    tree_root_map: List[int],
    debug: bool = True,
    stub_node_max: int = 50,
) -> List[int]:
    """
    Exclude tiny unbranched stubs and produce candidate indices.
    Keeps majors (>= 25% of max_nodes) or else keeps all non-stubs.
    """
    if not eligible_idx:
        return []
    max_nodes = max(feats[i]["node_count"] for i in eligible_idx)
    major_min_nodes = max(50, int(round(0.25 * max_nodes)))
    stubs, majors, minors = [], [], []
    for i in eligible_idx:
        m = feats[i]
        is_unbranched = (m.get("bifurcations", 0) == 0 and m.get("tip_count", 0) == 1)
        is_stub = is_unbranched and (m["node_count"] < stub_node_max)
        if is_stub:
            stubs.append(i)
            continue
        if m["node_count"] >= major_min_nodes:
            majors.append(i)
        else:
            minors.append(i)
    candidates = majors if majors else [i for i in eligible_idx if i not in stubs]
    if not candidates:
        candidates = list(eligible_idx)

    if debug:
        print(f"[apical-gate] max_nodes={max_nodes} major_min_nodes={major_min_nodes}")
        if stubs:
            print(f"[apical-gate] excluded_stubs: {[(i, tree_root_map[i], feats[i]['node_count']) for i in stubs]}")
        if minors:
            print(f"[apical-gate] excluded_minors: {[(i, tree_root_map[i], feats[i]['node_count']) for i in minors]}")
        print(f"[apical-gate] candidates: {[(i, tree_root_map[i], feats[i]['node_count']) for i in candidates]}")

    return candidates

def select_apical_tree_feature_fallback(
    graph: nx.DiGraph,
    trees: List[List[int]],
    eligible_idx: List[int],
    tree_root_map: List[int],
    soma_node: int,
    radial_step: float = 10.0,
    debug: bool = True,
) -> int:
    """
    A deterministic fallback (less complex) that scores candidates using a morphological heuristic.
    """
    feats: Dict[int, Dict[str, Any]] = {}
    for i in eligible_idx:
        feats[i] = compute_tree_features_aligned(graph, soma=soma_node, root=tree_root_map[i], tree_nodes=set(trees[i]), radial_step=radial_step)

    candidates = apical_candidate_filter(feats, eligible_idx, tree_root_map, debug=debug)

    def score(i: int) -> float:
        f = feats[i]
        s = 0.0
        s += math.log1p(f.get("max_path_length", 0.0)) * 1.3
        s += math.log1p(f.get("total_length", 0.0)) * 1.0
        s += math.log1p(f.get("sholl_sum", 0.0)) * 0.8
        s += math.log1p(f.get("node_count", 0.0)) * 0.6
        s += math.log1p(f.get("max_euclid", 0.0)) * 0.7
        # directional trunk signals
        axis_max = max(
            abs(float(f.get("principal_axis_x", 0.0))),
            abs(float(f.get("principal_axis_y", 0.0))),
            abs(float(f.get("principal_axis_z", 0.0))),
        )
        s += float(f.get("elongation", 0.0)) * 0.6
        s += float(axis_max) * 0.5
       
        return float(s)

    apical = max(candidates, key=score)

    if debug:
        print("\n[apical-fallback] Per-candidate scores:")
        for i in candidates:
            f = feats[i]
            axis_max = max(
                abs(float(f.get("principal_axis_x", 0.0))),
                abs(float(f.get("principal_axis_y", 0.0))),
                abs(float(f.get("principal_axis_z", 0.0))),
            )
            print(
                f"  idx={i} root={tree_root_map[i]} score={score(i):.3f} "
                f"[max_path={f.get('max_path_length',0.0):.2f}, max_euc={f.get('max_euclid',0.0):.2f}, "
                f"tot_len={f.get('total_length',0.0):.2f}, nodes={f.get('node_count',0)}, "
                f"sholl={f.get('sholl_sum',0.0):.2f}, elong={f.get('elongation',0.0):.3f}, axis_max={axis_max:.3f}]"
            )
        print(f"[apical-fallback] Selected APICAL idx={apical} root={tree_root_map[apical]}\n")

    return apical


# ------------------------------------------------------------
# CSV-free morphology selector with Sholl consideration
# ------------------------------------------------------------
def _local_z(values: List[float]) -> List[float]:
    """
    Z-score within current SWC across candidate trees.
    If std == 0, returns zeros to avoid division by zero.
    """
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return []
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if sd <= 1e-12:
        return [0.0] * len(values)
    return list((a - mu) / sd)


def select_apical_by_morphology(
    graph: nx.DiGraph,
    trees: List[List[int]],
    eligible_idx: List[int],
    tree_root_map: List[int],
    soma_node: int,
    radial_step: float = 10.0,
    debug: bool = True,
) -> int:
    """
    CSV-free apical selection using robust morphological features.
    Includes Sholl (sholl_sum).

    Composite score (per candidate):
      S = 1.5 * z(log1p(max_path_length))
        + 1.2 * z(log1p(max_euclid))
        + 0.8 * z(log1p(sholl_sum))
        + 0.7 * z(log1p(total_length))
        + 0.5 * z(log1p(node_count))
        + 0.8 * z(elongation)
        + 0.6 * z(max_abs(principal_axis))

    Tie-break by raw max_path_length, then raw max_euclid.
    """
    # Extract features
    feats: Dict[int, Dict[str, Any]] = {}
    for i in eligible_idx:
        feats[i] = compute_tree_features_aligned(
            graph,
            soma=soma_node,
            root=tree_root_map[i],
            tree_nodes=set(trees[i]),
            radial_step=radial_step,
        )

    # Gate obvious stubs/minors
    candidates = apical_candidate_filter(feats, eligible_idx, tree_root_map, debug=debug)
    if not candidates:
        candidates = list(eligible_idx)

    # Collect raw vectors for candidates
    max_path_vals, max_euc_vals = [], []
    sholl_vals, tot_len_vals, node_cnt_vals = [], [], []
    elong_vals, axis_max_vals = [], []

    for i in candidates:
        f = feats[i]
        max_path_vals.append(float(f.get("max_path_length", 0.0)))
        max_euc_vals.append(float(f.get("max_euclid", 0.0)))
        sholl_vals.append(float(f.get("sholl_sum", 0.0)))
        tot_len_vals.append(float(f.get("total_length", 0.0)))
        node_cnt_vals.append(float(f.get("node_count", 0.0)))
        elong_vals.append(float(f.get("elongation", 0.0)))
        axis_max_vals.append(max(
            abs(float(f.get("principal_axis_x", 0.0))),
            abs(float(f.get("principal_axis_y", 0.0))),
            abs(float(f.get("principal_axis_z", 0.0))),
        ))

    # Transform: log1p + z within SWC
    z_max_path = _local_z([math.log1p(x) for x in max_path_vals])
    z_max_euc  = _local_z([math.log1p(x) for x in max_euc_vals])
    z_sholl    = _local_z([math.log1p(x) for x in sholl_vals])
    z_tot_len  = _local_z([math.log1p(x) for x in tot_len_vals])
    z_node_cnt = _local_z([math.log1p(x) for x in node_cnt_vals])
    z_elong    = _local_z(elong_vals)
    z_axis_max = _local_z(axis_max_vals)

    # Weights (tune if needed)
    W = {
       "max_path": 1.5,
        "max_euc":  1.2,
        "sholl":    0.8,
        "tot_len":  0.7,
        "node_cnt": 0.5,
        "elong":    0.8,
        "axis_max": 0.6,
    }

    # Compute composite score
    idx_to_score: Dict[int, float] = {}
    for k, i in enumerate(candidates):
        score = (
            W["max_path"] * z_max_path[k]
            + W["max_euc"]  * z_max_euc[k]
            + W["sholl"]    * z_sholl[k]
            + W["tot_len"]  * z_tot_len[k]
            + W["node_cnt"] * z_node_cnt[k]
            + W["elong"]    * z_elong[k]
            + W["axis_max"] * z_axis_max[k]
        )
        idx_to_score[i] = float(score)

    # Pick argmax with tie-break
    best = max(candidates, key=lambda i: idx_to_score[i])
    best_score = idx_to_score[best]
    EPS = 1e-9
    near_ties = [i for i in candidates if abs(idx_to_score[i] - best_score) <= EPS]

    if len(near_ties) > 1:
        if debug:
            print(f"[apical-morph] Near tie among {near_ties}. Tie-break by max_path_length, then max_euclid.")
        def tie_key(i: int) -> Tuple[float, float]:
            f = feats[i]
            return (
                float(f.get("max_path_length", 0.0)),
                float(f.get("max_euclid", 0.0)),
            )
        best = max(near_ties, key=tie_key)

    if debug:
        print("\n[apical-morph] Candidate scores:")
        for i in candidates:
            f = feats[i]
            axis_max = max(
                abs(float(f.get("principal_axis_x", 0.0))),
                abs(float(f.get("principal_axis_y", 0.0))),
                abs(float(f.get("principal_axis_z", 0.0))),
            )
            print(
                f"  idx={i} root={tree_root_map[i]} score={idx_to_score[i]:.3f} "
                f"[max_path={f.get('max_path_length',0.0):.2f}, max_euc={f.get('max_euclid',0.0):.2f}, "
                f"sholl={f.get('sholl_sum',0.0):.2f}, tot_len={f.get('total_length',0.0):.2f}, "
                f"nodes={f.get('node_count',0)}, elong={f.get('elongation',0.0):.3f}, axis_max={axis_max:.3f}]"
            )
        print(f"[apical-morph] Selected APICAL idx={best} root={tree_root_map[best]} score={idx_to_score[best]:.3f}\n")

    return best


def assign_apical_basal_rule(
    graph: nx.DiGraph,
    radial_step: float = 10.0,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Compute per-tree features, select apical by morphology, retag nodes:
      - Apical tree -> node_type = 4
      - Other dendritic trees -> node_type = 3
      - Soma remains node_type = 1
    Returns summary dict.
    """
    soma_node = ensure_soma_node(graph)
    if soma_node is None:
        print("[rule] ERROR: No soma/root (parent==-1).")
        return {"apical_index": None, "apical_root": None, "changed_nodes": 0, "bad_basal_trees": 0}

    trees, tree_root_map = build_soma_children_trees(graph, soma_node)
    if not trees:
        print("[rule] No children trees under soma.")
        return {"apical_index": None, "apical_root": None, "changed_nodes": 0, "bad_basal_trees": 0}

    # Mark eligible (trees containing any dendritic type: 3 or 4)
    eligible_idx = []
    for i, tree_nodes in enumerate(trees):
        node_types = [_safe_int(graph.nodes[n].get("node_type", -1)) for n in tree_nodes]
        has_dendrite = any(nt in (3, 4) for nt in node_types)
        if has_dendrite:
            eligible_idx.append(i)
        if debug:
            print(f"[rule] Tree idx={i} root={tree_root_map[i]} eligible={has_dendrite} node_count={len(tree_nodes)}")

    if not eligible_idx:
        print("[rule] No dendritic trees found (type 3/4). Nothing to retag.")
        return {"apical_index": None, "apical_root": None, "changed_nodes": 0, "bad_basal_trees": 0}

    # --- Primary morphology-based selection ---
    apical_idx = select_apical_by_morphology(
        graph=graph,
        trees=trees,
        eligible_idx=eligible_idx,
        tree_root_map=tree_root_map,
        soma_node=soma_node,
        radial_step=radial_step,
        debug=debug,
    )

    # Fallback if something goes off
    if apical_idx is None:
        apical_idx = select_apical_tree_feature_fallback(
            graph=graph,
            trees=trees,
            eligible_idx=eligible_idx,
            tree_root_map=tree_root_map,
            soma_node=soma_node,
            radial_step=radial_step,
            debug=debug,
        )

    # retag
    changed = 0
    apical_set = {apical_idx}
    basal_set = set(eligible_idx) - apical_set

    # Basal: node_type = 3
    for i in basal_set:
        for n in trees[i]:
            if n == soma_node:
                continue
            cur = _safe_int(graph.nodes[n].get("node_type", -1))
            if cur != 3:
                graph.nodes[n]["node_type"] = 3
                changed += 1

    # Apical: node_type = 4
    for n in trees[apical_idx]:
        if n == soma_node:
            continue
        cur = _safe_int(graph.nodes[n].get("node_type", -1))
        if cur != 4:
            graph.nodes[n]["node_type"] = 4
            changed += 1

    # Diagnostics: basal trees must not have type=4 nodes
    bad_basal = 0
    for i in sorted(basal_set):
        bad_nodes = [n for n in trees[i] if _safe_int(graph.nodes[n].get("node_type", -1)) == 4]
        if bad_nodes:
            bad_basal += 1
            if debug:
                print("\n[WARN] Basal tree contains type=4 nodes (unexpected):")
                print(f"  tree_idx={i} root={tree_root_map[i]} count_type4={len(bad_nodes)} sample={bad_nodes[:10]}")

    if debug:
        print(f"\n[rule] DONE. changed_nodes={changed}, bad_basal_trees={bad_basal}")
        print(f"[rule] FINAL APICAL idx={apical_idx} root={tree_root_map[apical_idx]}\n")

    return {
        "apical_index": apical_idx,
        "apical_root": tree_root_map[apical_idx],
        "changed_nodes": changed,
        "bad_basal_trees": bad_basal,
    }

        

def _convert_tags_to_3_by_node_id_inplace(file_path, tag_set=TAG_SET, temp_tag=TEMP_TAG):
    """
    Convert node_type in {5,6,7} -> 3 (in-place).
    Returns restore_map: {node_id: original_type} for nodes changed.
    """
    restore_map = {}
    changed = False

    with open(file_path, "r", encoding="ISO-8859-1", errors="replace") as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue

        parts = line.split()
        if len(parts) < 7:
            out_lines.append(line)
            continue

        try:
            node_id = int(parts[0])
            node_type = int(parts[1])
        except Exception:
            out_lines.append(line)
            continue

        if node_type in tag_set:
            restore_map[node_id] = node_type
            parts[1] = str(temp_tag)
            out_lines.append(" ".join(parts) + "\n")
            changed = True
        else:
            out_lines.append(line)

    if changed:
        tmp = file_path + ".tmp_tagfix"
        with open(tmp, "w", encoding="ISO-8859-1") as f:
            f.writelines(out_lines)
        os.replace(tmp, file_path)

    return restore_map  


def _restore_tags_by_node_id_inplace(file_path, restore_map):
    """
    Restore node_type using restore_map {node_id: original_type} (in-place).
    """
    if not restore_map:
        return

    with open(file_path, "r", encoding="ISO-8859-1", errors="replace") as f:
        lines = f.readlines()

    out_lines = []
    changed = False

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue

        parts = line.split()
        if len(parts) < 7:
            out_lines.append(line)
            continue

        try:
            node_id = int(parts[0])
        except Exception:
            out_lines.append(line)
            continue

        if node_id in restore_map:
            parts[1] = str(int(restore_map[node_id]))
            out_lines.append(" ".join(parts) + "\n")
            changed = True
        else:
            out_lines.append(line)

    if changed:
        tmp = file_path + ".tmp_tagrestore"
        with open(tmp, "w", encoding="ISO-8859-1") as f:
            f.writelines(out_lines)
        os.replace(tmp, file_path)
        


def start_process():
    global process_running
    with process_lock:
        if process_running:
            return False
        process_running = True
        return True

def end_process():
    global process_running
    with process_lock:
        process_running = False


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

