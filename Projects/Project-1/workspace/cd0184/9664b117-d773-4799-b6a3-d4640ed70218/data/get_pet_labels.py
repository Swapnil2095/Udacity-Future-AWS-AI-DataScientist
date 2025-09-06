#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Swapnil Gaikwad
# DATE CREATED:  5th Sept 2025                                
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Replace None with the results_dic dictionary that you created with this
    # function
    # Create empty dictionary
    results_dic = dict()

    # Retrieve the filenames from folder
    filename_list = listdir(image_dir)

    # Process each filename
    for filename in filename_list:
        # Skip system/hidden files (like .DS_Store on Mac)
        if filename.startswith("."):
            continue

        # Convert filename to lowercase
        low_filename = filename.lower()

        # Split words by underscore
        word_list = low_filename.split("_")

        # Build pet name only from alphabetic words
        pet_name = ""
        for word in word_list:
            if word.isalpha():
                pet_name += word + " "

        # Strip extra whitespace from the final name
        pet_name = pet_name.strip()

        # Add to dictionary if not already there
        if filename not in results_dic:
            results_dic[filename] = [pet_name]
        else:
            print(f"** Warning: Duplicate file {filename} found in directory!")

    return results_dic


if __name__ == "__main__":
    test_dir = "pet_images/"
    results = get_pet_labels(test_dir)
    print(f"\nNumber of labels = {len(results)}\n")
    for i, (k, v) in enumerate(results.items()):
        print(f"{i+1:2d}. Filename: {k:30}  Label: {v[0]}")
        if i == 9:  # only show first 10
            break