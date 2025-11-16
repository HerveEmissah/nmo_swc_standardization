import os
import numpy as np
import re

def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                #list_of_results.append((line_number, line.rstrip()))
                list_of_results.append(line_number)
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results

#beginnning of fix_radius
def fix_radius(file_name, lineNumbers):
    os.chdir('../CNG_Version')
    file_name_swc = file_name[:len(file_name)-4]
    with open(file_name_swc) as flr:
            fls = flr.read().splitlines()
    i=0

    # Initialize fl array
    fl = np.empty((len(fls), 7), dtype=object)

    # Convert the string list to a numpy array and process valid lines
    for fll in fls:
        if fll.strip() and not fll.startswith("#"):  # Skip blank lines and lines starting with '#'
            X = fll.split()            
            X = np.array([int(X[0]), int(X[1]), float(X[2]), float(X[3]), float(X[4]), float(X[5]), int(X[6])])
                
            if len(X) == 6:
                X = np.append(X, [-1])  # Add -1 as the radius (7th element)
            
            if X[5] <= 0:  # X[5] refers to the radius (6th element in the array)
                X[5] = 0.5
                
            fl[i, :] = X
            i += 1
        else:
            # Keep the line unchanged if it starts with '#' or is blank
            fl[i, 0] = fll
            i += 1

    #delete variables
    del X, fls

    os.chdir("..")
    if not os.path.exists("out_radius"):
        os.mkdir("out_radius")
    os.chdir("out_radius")
   
    # Filter out lines that start with '#' or are blank and apply formatting to the rest
    formatted_rows = []
    for row in fl:
        row = ['' if item is None else item for item in row]
        if isinstance(row[0], str) and row[0].startswith("#"):
            formatted_rows.append([str(item) for item in row])
        elif any(row):  # If the row is not blank
            # Format the row as int, int, float, float, float, int
            formatted_row = [
                int(row[0]),
                int(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                int(row[6])
            ]
            formatted_rows.append(formatted_row)

    # Save the formatted lines to the file
    with open(file_name_swc, "w") as file:
        for formatted_row in formatted_rows:
            formatted_row_str = " ".join(map(str, formatted_row))
            file.write(formatted_row_str + "\n")

    #print(fl[:5])

    os.chdir("../Remaining_issues")


#main script
Erlist = search_string_in_file('Log.txt', '4.1=>')
ELi = np.array(Erlist)
ELi = np.insert(ELi, 0, 0., axis=0)

with open('Log.txt') as f:
    ids = f.read().splitlines()
substr = 'swc.std'
fnam = []
for i in range(1,len(ELi)) :
    for k in range(ELi[i]-1,ELi[i-1],-1):
        if substr in ids[k]:
            fnam.append(ids[k])
            break

del ELi, Erlist, ids, substr

output_radius_dir = '/nmo_swc/output_radius'
os.chdir('/nmo_swc/output_radius/Remaining_issues')

pattern = "4.1  Radius of line (.*?)"

for fn in fnam:
    file_path = os.path.join(output_radius_dir, fn)
    with open(fn) as f:
        Lines = f.read().splitlines()
    fids = search_string_in_file(fn, '4.1  Radius of line')
    #print (fids)
    lnum = []
    for fi in fids:
        str1 = Lines[fi-1]
        lineNum = re.search(pattern, str1).group(1)
        lnum.append(lineNum)
        #print(f"fi in {fn}: {fi}")  # Print the value of fi in each iteration
    try:
        fix_radius(fn, lnum)
        print("Fixed radius for file:", fn[:-4])
    except:
        os.chdir("../Remaining_issues")
        pass
    del Lines, fi, fids, lineNum, lnum, str1
    #clear variables


