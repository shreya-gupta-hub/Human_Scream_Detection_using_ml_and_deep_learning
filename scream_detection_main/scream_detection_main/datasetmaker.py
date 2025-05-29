import time
import numpy as np
import pandas as pd
import os
from scipy.io.wavfile import read

class scream():

    def adder(self):
        # Ensure necessary folders exist
        for folder in ['negative', 'positive', 'testing']:
            if not os.path.exists(folder):
                print(f"Folder '{folder}' does not exist. Creating it...")
                os.makedirs(folder)

        ################################  Getting Negative Sounds  ################################
        files = os.listdir('negative')
        arr = []
        for i in files:
            num = float(0)  # Assigning labels to negative sounds
            i = 'negative/' + i
            try:
                data, rs = read(i)
            except:
                print("Removed " + i)
                os.remove(i)
                continue
            try:
                rs = rs.astype(float)
                rs = np.insert(rs, 0, num)
                a = pd.Series(rs)
                arr.append(a)
                self.ctr += 1
            except:
                pass

        ################################  Getting Positive Sounds  ################################
        files = os.listdir('positive')
        for i in files:
            num = float(1)  # Assigning labels to positive sounds
            i = 'positive/' + i
            try:
                data, rs = read(i)
            except:
                print("Removed " + i)
                os.remove(i)
                continue
            try:
                rs = rs.astype(float)
                rs = np.insert(rs, 0, num)
                a = pd.Series(rs)
                arr.append(a)
                self.ctr += 1
            except:
                pass

        self.starting_index_not_to_be_shuffled = self.ctr

        # Save starting index for testing
        with open("beginning index of testing files.txt", "w") as file:
            file.write(str(self.ctr))

        print(str(self.ctr) + " have been added ")
        time.sleep(1)

        #################### Loading Testing Sounds #####################
        files = os.listdir('testing')
        for i in files:
            if i.startswith("1"):
                num = float(1)
            else:
                num = float(0)

            i = 'testing/' + i
            try:
                data, rs = read(i)
            except:
                print("Removed " + i)
                os.remove(i)
                continue
            try:
                rs = rs.astype(float)
                rs = np.insert(rs, 0, num)
                a = pd.Series(rs)
                arr.append(a)
                self.ctr += 1
            except:
                pass

        ########################## Final Data Preparation ##########################
        df = pd.DataFrame(arr)
        df = df.dropna(axis=1)  # Remove columns containing null or NA
        df.to_csv('resources.csv')

    def __init__(self):  # Fixed the typo from _init_ to __init__
        self.ctr = 0
        self.starting_index_not_to_be_shuffled = 0
        self.adder()
        start_time = time.time()
        print('Started processing...')

        if os.path.exists('resources.csv'):
            self.df = pd.read_csv('resources.csv', index_col=0, engine='c')
            print("Without shuffling, dataset contains " + str(len(self.df)) + " rows and " + str(len(self.df.columns)) + " columns")
        else:
            print("Error: 'resources.csv' not found. Please check the adder method.")
            return

        self.df.iloc[:self.starting_index_not_to_be_shuffled, :] = self.df.iloc[:self.starting_index_not_to_be_shuffled, :].sample(frac=1).reset_index(drop=True)  # Shuffle dataframe
        print("After shuffling, dataset contains " + str(len(self.df)) + " rows and " + str(len(self.df.columns)) + " columns")

        # Save shuffled dataset
        try:
            self.df.to_csv('newresources.csv')
            print("'newresources.csv' has been saved successfully.")
        except Exception as e:
            print(f"Error while saving 'newresources.csv': {e}")

        with open("input dimension for model.txt", "w") as file:
            file.write(str(len(self.df.columns) - 1))

        print("\nWhole process took %s seconds" % (time.time() - start_time))


scream()
