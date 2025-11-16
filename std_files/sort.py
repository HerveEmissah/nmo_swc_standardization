import os
import glob
import numpy as np

try:
    # Change directory to "SWC"
    os.chdir("SWC")
except FileNotFoundError:
    print("Error: The directory 'SWC' does not exist.")
    exit(1)
except Exception as e:
    print(f"Error changing directory: {e}")
    exit(1)

filter = "*.swc"
all_f = glob.glob(filter)

if not all_f:
    print("No SWC files found in the directory.")
    exit(0)

for f in all_f:
    try:
        print("SORTING FILE: " + f)
        
        # Load the file, ensuring proper format and handling empty files
        try:
            fl = np.loadtxt(f, dtype=float, comments="#")
            if fl.size == 0:
                print(f"Warning: File {f} is empty. Skipping.")
                continue
        except ValueError:
            print(f"Error: File {f} has incorrect format. Skipping.")
            continue
        except Exception as e:
            print(f"Error reading file {f}: {e}. Skipping.")
            continue

        # Initialize sorted neuron structure
        sNeu = np.empty((0, 7), float)
        Px = np.where(fl[:, 6] == -1)
        Px = list(Px[0])
        
        while len(Px) > 0:
            try:
                P = Px[0]
                Px = Px[1:]
                while P.size > 0:
                    P = int(P)
                    sNeu = np.vstack((sNeu, fl[P, :]))
                    Child = np.where(fl[:, 6] == fl[P, 0])
                    Child = list(Child[0])
                    if len(Child) == 0:
                        break
                    if len(Child) > 1:
                        Px = np.append(Child[1:], Px)
                    P = Child[0]
            except Exception as e:
                print(f"Error during processing file {f}: {e}")
                break

        try:
            # Update parent ID references
            sRe = sNeu[:, 6]
            Li = list(range(1, (len(sNeu[:, 1]) + 1)))
            Li1 = Li[:-1]
            for i in Li1:
                if sNeu[i, 6] != -1:
                    pids = np.where(sNeu[:, 0] == sNeu[i, 6])
                    pids = float(pids[0])
                    sRe[i] = pids + 1
            sNeu[:, 6] = sRe
            sNeu[:, 0] = Li
        except Exception as e:
            print(f"Error updating parent IDs in file {f}: {e}")
            continue

        # Save the sorted data back to the file
        try:
            np.savetxt(f, sNeu, fmt="%u %u %f %f %f %f %d")
            print(f"File {f} sorted and saved successfully.")
        except Exception as e:
            print(f"Error saving file {f}: {e}")
    except Exception as e:
        print(f"Unexpected error with file {f}: {e}")

print("Processing completed.")
