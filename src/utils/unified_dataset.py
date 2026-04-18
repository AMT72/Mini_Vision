#######################################################################################################################
#--------------------Essentails: IMPORTING LIBRARIES (MAKE SURE PIP REQUIREMENT ARE INSTALLED) !!!--------------------#
#######################################################################################################################
import sys

try:
    import os
    import json
    import shutil
    from PIL import Image
    from natsort import natsorted # Essential for natural ordering (1, 2, 10... instead of 1, 10, 2)
    from master_mapper import master_mapper
    from tqdm import tqdm #shows progress bar
    import gdown    #download from google drive


except ModuleNotFoundError as e:
    print(f"\n❌ ERROR: Missing dependency -> {e.name}. maybe some packages are missing?")
    print(" make sure to run: pip install -r requirements.txt")
    sys.exit(1) #if packages are missing , terminate the program


# ALL THE PATHS ARE HERE (You shoulnd't have to change anything if you followed the structure)

#images folders
damage_img_dir = "data/Car damages dataset/File1/img"
parts_img_dir =  "data/Car parts dataset/File1/img"
healthy_img_dir = "data/Car healthy dataset/img"
#put all the folders in a list:
source_dir = [damage_img_dir, parts_img_dir,healthy_img_dir]

# the output path:
output_img_dir = "data/unified_dataset/images"

#make the directory if it not there:
os.makedirs(output_img_dir, exist_ok=True)


# Paths for JSON sources
damage_ann_dir = "data/Car damages dataset/File1/ann"
parts_ann_dir = "data/Car parts dataset/File1/ann"
output_labels_dir = "data/unified_dataset/labels"
#put them in a list
ann_source_dir = [damage_ann_dir, parts_ann_dir]


#############################################################################
#--------------------DOWNLOAD THE DATASET (FROM G-DRIVE)--------------------#
#############################################################################

def download_healthy_data():
    file_id = "1sBGUA0Xg16epBb2qHKcOSxCN1D7T1-3e"
    url = f"https://drive.google.com/uc?id={file_id}"

    #where the data will be unzipped
    base_dir = 'data'
    zip_path = os.path.join(base_dir, "Car healthy dataset.zip")

    # Path after extracting
    final_dir= os.path.join(base_dir, "Car healthy dataset")
    
    #start extracting
    if not os.path.exists(final_dir):
        os.makedirs(final_dir, exist_ok=True)
        print(f'Downloading Healthy dataset, please wait. . .')

        gdown.download(url,zip_path , quiet = False)

        #extract it
        shutil.unpack_archive(zip_path, final_dir)

        #remove the zip after extracting:
        os.remove(zip_path)
        print("Done, data is ready")

    else:
        print(f"Data already exists, skipping Download")


#############################################################################
#--------------------IMAGES_CONVERTION_FUNCTION (TO_PNG)--------------------#
#############################################################################

#start converting to png:

# Dictionary to link old names to new names (IMPORTANT for JSON stage)
name_mapping = {}

def convert_to_png(source_dir,output_dir,prefix="Vehicle"):
    #counter
    processed_count = 0 

    #if the file already exist , remove it first:
    if os.path.exists(output_img_dir):
        print(f"found files in ({output_dir}), cleaning...")
        shutil.rmtree(output_img_dir)
    os.makedirs(output_img_dir, exist_ok=True)

    # Tuple of valid extentions:
    valid_extensions = ('.jpg', '.webp', '.jpeg','.png')

    #create a set to store the paths (without duplicates):
    unique_files = set()

    #A dictionary to remeber which file belongs to which folder (did the image come from parts or damage imgs)
    file_location = {}
    


    # check each folder and link its image with it
    for folder in source_dir:
        if not os.path.exists(folder) :
            continue #if the folder does not exist move on to the next folder

        for filename in os.listdir(folder):
            if filename.lower().endswith(valid_extensions):
                if filename not in unique_files:
                    unique_files.add(filename)
                    file_location[filename] = folder

    

    for filename in tqdm(natsorted(unique_files), desc = "converting images to png (unified_dataset/images)"):
            processed_count += 1

            # Get full original path
            full_path = os.path.join(file_location[filename],filename)


            #opne the image
            img = Image.open(full_path).convert('RGB')
            
            #make a new name for it:
            new_filename = f"{prefix}_{processed_count:04d}.png"
            target_path = os.path.join(output_dir, new_filename)

            img.save(target_path, "png")

            # Store the mapping (Old Name -> New Name)
            # We store only the stem (name without extension)
            old_stem = os.path.splitext(filename)[0]     # Car damages 2 (without the extention)
            new_stem = os.path.splitext(new_filename)[0]  # what Car damages 2 maps to -> eg. Vehicle_0001
            name_mapping[old_stem] = new_stem              # {"Car damages 2": "Vehicle_0001"}

            pass # end of .endswith if statement


    return processed_count,name_mapping #end of the function


#########################################################################
#--------------------JSON -> TXT CONVERTION_FUNCTION--------------------#
#########################################################################

def normalize_annotation(source_dir, mapping_dic, mapper , output_dir):

    #Counter:
    processed = 0


    images_with_labels = set()

    # Clean output labels dir once at the start
    if os.path.exists(output_dir):
        print(f"found files in{output_dir}, cleaning...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for folder in source_dir:
        if not os.path.exists(folder):
            continue

        for filename in tqdm(natsorted(os.listdir(folder)), desc = "Generating .txt files to (unified_dataset/labels):"):
            if filename.endswith('.json'):

                #stript the name from the extenstions 
                clean_name = os.path.splitext(os.path.splitext(filename)[0])[0]

                #if the name exist -> map it to the new name (Car damages 2 maps to -> Vehicle_0001)
                if clean_name in mapping_dic:
                    new_name = mapping_dic[clean_name]
                    images_with_labels.add(new_name) # Track this image


                    #open the json file:
                    json_path = os.path.join(folder,filename)
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                        img_height, img_wdith = data['size']['height'] , data['size']['width']

                        #start Normalize 

                        #list to hold the coordinates + class -> [3 0.534523452 0.35123412 ...etc]
                        yolo_lines = []

                        for obj in data['objects']:
                            class_name = obj['classTitle'] # important for later to be used with the master mapper

                            #list to hold nomralized yolo cords (to later be added to yolo_lines)
                            yolo_coords = []


                            #get the normalized values 
                            for coordinates in obj['points']['exterior']:
                                x_norm = round(coordinates[0] / img_wdith, 8)
                                y_norm = round(coordinates[1]/ img_height, 8)

                                #put the coords in the yolo_coords
                                yolo_coords.extend([str(x_norm) , str(y_norm)])
                                
                            #apend the class + coords (speperated by space " ")
                            mapped_class = mapper[class_name]

                            yolo_line = f"{mapped_class} " + " ".join(yolo_coords) + "\n"

                            yolo_lines.append(yolo_line)

                        #now we have everything, just need to make txt files:
                        txt_name = f"{new_name}.txt"
                        output_path = os.path.join(output_dir , txt_name)

                        #start writing to the txt files: (note that we use append not write to be able to add overlapping information to the txt file in case there was any)
                        with open(output_path, 'a') as out_f:
                            # Join all lines found in this JSON and write them
                            out_f.writelines(yolo_lines)

                    # Increment the counter after a successful file process
                    processed += 1

    # 2. THE HEALTHY CARS LOGIC: Create empty files for remaining images
    # Every image in mapping_dic must have a corresponding .txt file
    for old_name, new_name in mapping_dic.items():
        if new_name not in images_with_labels:
            # Create an empty .txt file for background samples
            open(os.path.join(output_dir, f"{new_name}.txt"), 'a').close()
            # print(f"Created empty label for healthy car: {new_name}")

        
    return processed


#######################################################################################
#--------------------Calling THE FUNCTIONS (running the program) --------------------#
######################################################################################

if __name__ == "__main__":

    #download the data if its not there !
    download_healthy_data()


    # Execute Stage 1 (convert_to_png)
    total_images, mapping_dic =  convert_to_png(source_dir=source_dir,output_dir=output_img_dir)
    print(f"1st part DONE: Processed {total_images} Images")



    #Execute Stage 2 (txt generation)
    total_labels = normalize_annotation(
        source_dir=ann_source_dir, 
        mapping_dic=mapping_dic, 
        mapper=master_mapper, 
        output_dir=output_labels_dir
    )

    print(f"2nd part DONE: {total_labels} JSON files processed and unified!")
    print(f"Final dataset ready in: data/unified_dataset/") 