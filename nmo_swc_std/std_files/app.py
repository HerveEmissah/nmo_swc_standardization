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
import shutil, os
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

import sys
import glob


import torch
import torch.nn as nn
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

UPLOAD_FOLDER = 'Source-Version'  # Folder to store uploaded files
DOWNLOAD_FOLDER = 'downloads'  # Folder to serve downloadable files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

# clear log content if exists
file_path = '/nmo_swc/log/app.log'
if os.path.exists(file_path):
  # Open the file in write mode, which clears its contents
  with open(file_path, 'w') as file:
    pass

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

@app.route('/nmo/SWC_STD', methods=["GET", "POST"])
def SWC_STD():

    response = Response(status=200)

    # clear log content if exists
    file_path = '/nmo_swc/log/app.log'
    if os.path.exists(file_path):
       # Open the file in write mode, which clears its contents
       with open(file_path, 'w') as file:
          pass

       result = SWC_LONG_CONNECTIONS_STD()

       #if "error_code" not in result:
       # Define the path to Measurement
       print('\n***Starting Measurements...')

       # Clear Archives2process folder
       os.system(f'rm -rf /nmo_swc/Archives2process/*')

       # Clear output_Final
       os.system('rm -rf /nmo_swc/output_Final/*')
       
       Measurement_dir = f'/nmo_swc/Archives2process/{archive_folder_name}'
       os.makedirs(Measurement_dir)
       os.chmod(Measurement_dir, 0o777)
       
       CNGVersion_dir = f'/nmo_swc/Archives2process/{archive_folder_name}/CNGVersion'
       os.makedirs(CNGVersion_dir)
       os.chmod(CNGVersion_dir, 0o777)
       
       Measurements_dir = f'/nmo_swc/Archives2process/{archive_folder_name}/Measurements'
       os.makedirs(Measurements_dir)
       os.chmod(Measurements_dir, 0o777)

       #sys.exit()  #exit here for testing
       
       CNG_dir = '/nmo_swc/out_Final/CNG_Version/'
       # Loop through all files in the directory and remove those with size 0
       for filename in os.listdir(CNG_dir):
           file_path = os.path.join(CNG_dir, filename)
    
           # Check if it is a file and has a size of 0 bytes
           if os.path.isfile(file_path) and os.stat(file_path).st_size == 0:
               file_name = os.path.basename(file_path)
               print(f"Not able to Process file: {file_name}")
               os.remove(file_path)

       # process the files only if CNG_Version dir is not empty 
       if not os.listdir(CNG_dir):
           out_dir = '/nmo_swc/out_Final/'
           os.chdir(out_dir)       
           os.system('mv CNG_Version \'CNG Version\'')

           output_dir = f'/nmo_swc/output_Final/{archive_folder_name}_Final'
           if not os.path.exists(output_dir):
               os.makedirs(output_dir)
               os.chmod(output_dir, 0o777)

       else:
           os.system(f'cp /nmo_swc/out_Final/CNG_Version/*.swc {CNGVersion_dir}/')
           #os.system(f"cp '/nmo_swc/out_Final/'CNG Version'/*.swc {CNGVersion_dir}/")
           os.system(f'chmod 777 {CNGVersion_dir}/*.*')
       
           working_dir = '/nmo_swc'         
           os.chdir(working_dir)
       
           os.system('sh Run_LMProcess.sh')
       
           os.system(f'cp {Measurement_dir}/Measurements/*.* /nmo_swc/out_Final/Measurements/')
           os.system(f'rm -rf {Measurement_dir}/Measurements')
           os.system(f'rm -rf {Measurement_dir}/CNGVersion/*')

           # Copy LMProcessLog.txt to out_Final and remove file from Archives2process
           os.system(f'cp /nmo_swc/Archives2process/*.txt /nmo_swc/out_Final/')

           os.system(f'rm -rf /nmo_swc/Archives2process/*.txt')

           # Define the path to PNG Automation
           print('\n***PNG Automation In Progress...')
           PNG_dir = '/nmo_swc/PNG_Automation_SN/PNG_Automation_SN'
           #os.chdir(PNG_dir)
           os.chdir(working_dir)
           os.system(f'rm -rf {PNG_dir}/PNG/*.*')
           os.system(f'rm -rf {PNG_dir}/SWC/*.*')
           #os.system('chmod -R 755 /nmo_swc/out_Final/CNG_Version')
           #os.system("chmod -R 755 /nmo_swc/out_Final/\"CNG Version\"")
           os.system(f'cp /nmo_swc/out_Final/CNG_Version/*.swc {PNG_dir}/SWC/')
           #os.system(f"cp '/nmo_swc/out_Final/'CNG Version'/*.swc {PNG_dir}/SWC/")
           
           #os.system('sh PNG_generator.sh')           

           if not is_tomcat_running():
              print("Tomcat is not running. Starting Tomcat...")
              os.system("/opt/tomcat/bin/catalina.sh start")

              # Wait until Tomcat is ready
              print("Waiting for Tomcat to complete startup...")
              max_wait = 60  # seconds
              start_time = time.time()
              while not is_tomcat_running():
                 if time.time() - start_time > max_wait:
                    print("Tomcat is not running. Cannot generate PNG...")
                    raise RuntimeError("Tomcat failed to start within 60 seconds")
                 time.sleep(2)
              print("Tomcat is up and running!")
           else:
              print("Tomcat is already running.")
              
           os.system('python3 process_swcs.py')

           Img_dir = '/nmo_swc/out_Final/Images'
           if not os.path.exists(Img_dir):
             os.makedirs(Img_dir)
           os.chmod(Img_dir, 0o777)

           os.system(f'mkdir /nmo_swc/out_Final/Images/PNG')
           os.system(f'cp {PNG_dir}/PNG/*.png /nmo_swc/out_Final/Images/PNG/')

           os.system(f'rm -rf {PNG_dir}/PNG/*.png')

           # rename folders in out_Final
           out_dir = '/nmo_swc/out_Final/'
           os.chdir(out_dir)
       
           os.system('mv CNG_Version \'CNG Version\'')
           os.system('mv Remaining_issues \'Remaining issues\'')
           os.system('mv /nmo_swc/Standardizationlog \'Standardization log\'')

           # Copy all files from out_Final to mounted output_Final
           os.system(f'mkdir /nmo_swc/output_Final/{archive_folder_name}_Final')
           output_dir = f'/nmo_swc/output_Final/{archive_folder_name}_Final'
           os.chmod(output_dir, 0o777)
           os.system(f'cp -r /nmo_swc/out_Final/* /nmo_swc/output_Final/{archive_folder_name}_Final/')

           # Move all .std files from Normalized/Possible-issues to {archive_folder_name}_Final/Possible-issues
           os.system(f'mkdir /nmo_swc/output_Final/{archive_folder_name}_Final/Possible-issues')

           os.system('chmod 777 /nmo_swc/Normalized/Possible-issues/*.std')
           os.system(f'cp /nmo_swc/Normalized/Possible-issues/*.std /nmo_swc/output_Final/{archive_folder_name}_Final/Possible-issues/')

           # Move all log files to {archive_folder_name}_Final
           os.system(f'mv /nmo_swc/Normalized/Possible-issues/Log.txt /nmo_swc/output_Final/{archive_folder_name}_Final/Log1.txt')
           os.system(f'mv /nmo_swc/output_Final/{archive_folder_name}_Final/\'Remaining issues\'/Log.txt /nmo_swc/output_Final/{archive_folder_name}_Final/Log2.txt')
       
       # Copy all original swc files to Source-Version
       os.system(f'mkdir /nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version')
       upload_folder_path = f'/nmo_swc/{UPLOAD_FOLDER}'
       os.system(f'cp {upload_folder_path}/* /nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version/')

       # Get lists of files in each directory to Generate the final control file
       source_version_dir = f'/nmo_swc/output_Final/{archive_folder_name}_Final/Source-Version'
       cng_version_dir = os.path.join('/nmo_swc/output_Final', f'{archive_folder_name}_Final', 'CNG Version')
       control_file = f'/nmo_swc/output_Final/{archive_folder_name}_Final/control.txt'

       source_files = get_files_list(source_version_dir)
       cng_files = get_files_list(cng_version_dir)

       # Print the lists of files for debugging
       #print("Source files:", source_files)
       #print("CNG files:", cng_files)

       #missing_files = [file for file in source_files if file not in cng_files]

       # Get file names without extensions
       source_files_without_ext = []
       for file in source_files:
          if file.endswith('.swc'):
             try:
                filename_without_ext = get_filename_without_extension(file)
                source_files_without_ext.append(filename_without_ext)
                #print(f"source file: {filename_without_ext}")
             except Exception as e:
                print(f"Error getting source file: {file} - {e}")

       cng_files_without_ext = []
       for file in cng_files:
         if file.endswith('.CNG.swc'):
            try:
               filename_without_ext = get_filename_without_extension(file)
               cng_files_without_ext.append(filename_without_ext)
               #print(f"Processed CNG file: {filename_without_ext}")
            except Exception as e:
               print(f"Error with processing CNG file: {file} - {e}")


       # Find missing files
       try:
          missing_files = [file for file in source_files_without_ext if file not in cng_files_without_ext]
          #print("Missing files:", missing_files)
       except Exception as e:
          print(f"Error finding missing files: {e}")

       # Write result to control file
       try:
          with open(control_file, "w") as f:
             f.write(f"Total files in Source-version folder: {len(source_files)}\n")
             f.write(f"Total processed files in CNG-Version folder: {len(cng_files)}\n\n")
             f.write("Files in Source-version not processed:\n")
             for file in missing_files:
                f.write(f"{file}.swc\n")
          #print("Control file created:", control_file)
       except Exception as e:
          print(f"Error writing to control file: {e}")

       print('\n**SWC STANDARDIZATION COMPLETED**')

    else:
       print('\n**NO SWC FILE STANDARDIZED**')

    # Clear upload_dir '/nmo_swc/Source-Version'
    upload_dir = '/nmo_swc/Source-Version'
    clear_folder(upload_dir)

    return '**SWC STANDARDIZATION COMPLETED**'

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
    global  archive_folder_name
    try:
        # Check if the POST request contains files
        if 'files' not in request.files:
            print ('No files uploaded')
            return jsonify({'message': 'No files uploaded'}), 400

        files = request.files.getlist('files')

        if not files:
            print ('***No files uploaded***')
            return jsonify({'message': 'No files uploaded'}), 400
        else:
            print('a')
            folder_path = files[0].filename.rsplit('/', 1)[0]
            print('folder_path: ' + folder_path)
            archive_folder_name = folder_path.rsplit('/', 1)[-1]
            # Remove blanks from folder name
            archive_folder_name = archive_folder_name.replace(' ', '')
            print('archive_folder_name')

        # Create the uploads folder if it doesn't exist
        #if not os.path.exists(UPLOAD_FOLDER):
        #    print(f"The UPLOAD FOLDER '{UPLOAD_FOLDER}' did not exist. Creating the path...")
        #    wkdir = os.getcwd()
        #    print ('Working Directory: ', wkdir)
        #    os.makedirs(UPLOAD_FOLDER)

        # clear log content if exists
        file_path = '/nmo_swc/log/app.log'
        if os.path.exists(file_path):
            # Open the file in write mode, which clears its contents
            with open(file_path, 'w') as file:
                pass

        # Empty the upload folder before saving new files
        #empty_folder(app.config['UPLOAD_FOLDER'])
        upload_folder_path = f'/nmo_swc/{UPLOAD_FOLDER}'
        print (' ')
        empty_folder(upload_folder_path)

        curr_wrkdir = '/nmo_swc'
        os.chdir(curr_wrkdir)

        # Save each uploaded file
        print('ARCHIVE NAME:' + archive_folder_name)
        print(' ')
        for file in files:
            filename_only = os.path.basename(file.filename)
            print ('Saving ' + filename_only)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename_only)
            file.save(file_path)
            print (filename_only + ' uploaded and saved in CNG Server\n')
            os.chmod(file_path, 0o777)


        return jsonify({'message': 'Files uploaded successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/nmo/download', methods=['GET'])
def download():
    try:
        print('\n\n***Downloading...')
        timestamp = request.args.get('timestamp', '')

        # Mounted container output folder
        #output_folder = '/nmo_swc/output_Final'
        output_folder = '/nmo_swc/out_Final'
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
                            print(f'Added {file_path} to the archive')
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
        response.headers['Content-Disposition'] = 'attachment; filename=swc_corrected_tags_zips.zip'

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

    response = Response(status=200)
    
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
        source_version_dir = f'/nmo_swc/Source-Version'
        cng_dir = f'/nmo_swc/out'
        control_file = f'/nmo_swc/output/control.txt'

        # Get the list of files in each directory
        source_files = get_files_list(source_version_dir)
        cng_files = get_files_list(cng_dir)

        # Get file names without extensions
        source_files_without_ext = []
        for file in source_files:
           if file.endswith('.swc'):
              try:
                 filename_without_ext = get_filename_without_extension(file)
                 source_files_without_ext.append(filename_without_ext)
                 #print(f"source file: {filename_without_ext}")
              except Exception as e:
                 print(f"Error getting source file: {file} - {e}")

        cng_files_without_ext = []
        for file in cng_files:
          if file.endswith('.CNG.swc'):
             try:
                filename_without_ext = get_filename_without_extension(file)
                cng_files_without_ext.append(filename_without_ext)
                #print(f"Processed CNG file: {filename_without_ext}")
             except Exception as e:
                print(f"Error with processing CNG file: {file} - {e}")

        # Find missing files
        try:
           missing_files = [file for file in source_files_without_ext if file not in cng_files_without_ext]
           #print("Missing files:", missing_files)
        except Exception as e:
           print(f"Error finding missing files: {e}")

        # Write result to control file
        try:
           with open(control_file, "w") as f:
              f.write(f"Total files in Source-version folder: {len(source_files)}\n")
              f.write(f"Total processed files: {len(cng_files)}\n\n")
              f.write("Files in Source-version not processed:\n")
              for file in missing_files:
                 f.write(f"{file}.swc\n")
           #print("Control file created:", control_file)
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

    return response


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

       # Run Finalize
       os.system(f'java -jar Finalize.jar {dir_name}')     

       # Run Check to generate log file
       os.system('java -jar Check.jar')

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

       # Run Finalize
       working_dir = '/nmo_swc'
       os.chdir(working_dir)
       out_dir_name = 'out'
       os.system(f'java -jar Finalize.jar {out_dir_name}')

       # Run Check again to generate log file
       os.system('java -jar Check.jar')
       
      
@app.route('/nmo/CorrectTag', methods=["GET", "POST"])
def CorrectTag():

    response = Response(status=200)

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

    print('\n***Processing Correction of mislabeled pyramidals...\n')
    # Load model
    model = GCN(input_dim=3, hidden_dim=16, output_dim=2)
    #model.load_state_dict(torch.load("GCN_model.pth"))
    state_dict = torch.load("GCN_model.pth", weights_only=True)  # Explicitly set weights_only=True for future-proofing
    model.load_state_dict(state_dict)
    model.eval()

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
    for file in files:
        #print(f"Reading file {file}")
        #graph = read_swc_file(file)

        swc_filename = os.path.basename(file)
        print(f"Processing & Correcting Tag for file {swc_filename}")
        os.system(f'python3 correct_pyramidal_tags.py {file} n')
        
        #predict_and_correct_node_types(graph, model)        
        #print(f"Saving file {file}")
        #save_swc_file(graph, output_path)

        output_path = os.path.join(out_dir_name, os.path.basename(file))
        print(f"Corrected file saved at {output_path}")
        print('')
    
    print('\n***PNG Automation In Progress...')
    PNG_dir = '/nmo_swc/PNG_Automation_SN/PNG_Automation_SN'
    os.chdir(PNG_dir)
    os.system(f'rm -rf {PNG_dir}/PNG/*.*')
    os.system(f'rm -rf {PNG_dir}/SWC/*.*')
    os.system(f'cp /nmo_swc/{out_dir_name}/*.swc {PNG_dir}/SWC/')
    #os.system('sh PNG_generator.sh')

    wrk_dir = '/nmo_swc/'
    os.chdir(wrk_dir)

    if not is_tomcat_running():
       print("Tomcat is not running. Starting Tomcat...")
       os.system("/opt/tomcat/bin/catalina.sh start")

       # Wait until Tomcat is ready
       print("Waiting for Tomcat to complete startup...")
       max_wait = 60  # seconds
       start_time = time.time()
       while not is_tomcat_running():
          if time.time() - start_time > max_wait:
             print("Tomcat is not running. Cannot generate PNG...")
             raise RuntimeError("Tomcat failed to start within 60 seconds")
          time.sleep(2)
       print("Tomcat is up and running!")
    else:
       print("Tomcat is already running.")

    os.system('python3 process_swcs.py')
    
    os.system(f'mv {PNG_dir}/PNG/*.png /nmo_swc/{out_png_dir_name}/')

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
    os.system(f'mv /nmo_swc/{out_png_dir_name}/*.* {out_png_dir_name_mv}/')
    
    print('\n**PNG GENERATION COMPLETED**')
    print('\n**SWC TAG CORRECTION COMPLETED**')

    return '**SWC TAG CORRECTION COMPLETED**'


# SWC file reading function
def read_swc_file(filename):
    swc_df = pd.read_csv(filename, sep=r'\s+', comment='#', header=None,
                         names=['node_id', 'node_type', 'x', 'y', 'z', 'radius', 'parent'],
                         encoding='ISO-8859-1')
    swc_df['node_id'] = swc_df['node_id'].astype(int)
    swc_df['node_type'] = swc_df['node_type'].astype(int)
    swc_df['x'] = swc_df['x'].astype(float)
    swc_df['y'] = swc_df['y'].astype(float)
    swc_df['z'] = swc_df['z'].astype(float)
    swc_df['radius'] = swc_df['radius'].astype(float)
    swc_df['parent'] = swc_df['parent'].astype(int)

    G = nx.DiGraph()
    for _, row in swc_df.iterrows():
        G.add_node(row['node_id'], node_type=int(row['node_type']), x=row['x'], y=row['y'], z=row['z'])
        if row['parent'] != -1:
            G.add_edge(row['parent'], row['node_id'])
    return G

# Prediction and correction function
def predict_and_correct_node_types(graph, model):
    node_features = np.array([[graph.nodes[node]['x'], graph.nodes[node]['y'], graph.nodes[node]['z']] for node in graph.nodes])
    node_tensor = torch.tensor(node_features, dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(node_tensor)
        _, predicted_labels = torch.max(predictions, 1)
        
    for i, node in enumerate(graph.nodes):
        graph.nodes[node]['node_type'] = int(3 if predicted_labels[i] == 0 else 4)


# Function to save SWC file
def save_swc_file(graph, filename):
    with open(filename, 'w') as f:
        for node in graph.nodes:
            data = graph.nodes[node]
            parent = next(graph.predecessors(node), -1)
            f.write(f"{int(node)} {int(data['node_type'])} {data['x']} {data['y']} {data['z']} 1.0 {int(parent)}\n")



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

