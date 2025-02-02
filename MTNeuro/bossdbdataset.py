# dataset class for our dataset which is hosted on bossdb
# the array function comes from intern, which we use to interface with bossdb
# This dataset supports up to 4 regions, but each region must have the same size z,y,x cutout. 
from __future__ import annotations
from pyrsistent import m
from torch.utils.data import Dataset
from intern import array
import numpy as np
import torch
import matplotlib 
from os.path import exists
from requests.exceptions import HTTPError
from torchvision import transforms
import warnings

class BossDBDataset(Dataset):
    """
    Downloads the data from BossDB using array() method from intern library. Further the objects of this class can be used to access the downloaded data slices/volumes by index.
    """
    def __init__(
        self, 
        task_config: dict,
        boss_config: dict=None,
        mode="train",
        image_transform=None,
        mask_transform=None,
        transform=None,
        retries = 5,
        download = True,
        download_path = './'
    ):
        """
        Downloads the data, calculates the centroids and initializes the necessary variables.
        
        Args:
            task_config:        The JSON config file with the required task configurations.
            boss_config:        The config file to pass to the array() method from intern library.
            mode:               Can be set to "train", "test" or "val" to indicate which set to make available.
            image_transform:    The transform specified will be applied on the downloaded slices before it is returned.
            mask_transform:     (deprecated) The transform to apply to the annotations.
            transform:          The transform applied to both image and mask.
            retries:            The number of times to retry if the connection attempt to BossDB fails.
            download:           If set to 'True', pre-downlaoded data will be used, else if it's not available, the data will be downloaded & saved.
            download_path:      The location to download the data to / access it from.

        Returns:
            -
        """
        #Calculate the centroids for the slices along x and y for the cortex region. 
        x_cor = np.arange(task_config["tile_size"][0]/2, task_config["xrange_cor"][1]-task_config["xrange_cor"][0] ,task_config["tile_size"][0])
        y_cor = np.arange(task_config["tile_size"][1]/2, task_config["yrange_cor"][1]-task_config["yrange_cor"][0] ,task_config["tile_size"][1])

        #Calculate the centroids for the slices along x and y for the striatum region. 
        x_stri = np.arange(task_config["tile_size"][0]/2, task_config["xrange_stri"][1]-task_config["xrange_stri"][0] ,task_config["tile_size"][0])
        y_stri = np.arange(task_config["tile_size"][1]/2, task_config["yrange_stri"][1]-task_config["yrange_stri"][0] ,task_config["tile_size"][1])

        #Calculate the centroids for the slices along x and y for the VP region.
        x_vp = np.arange(task_config["tile_size"][0]/2, task_config["xrange_vp"][1]-task_config["xrange_vp"][0] ,task_config["tile_size"][0])
        y_vp = np.arange(task_config["tile_size"][1]/2, task_config["yrange_vp"][1]-task_config["yrange_vp"][0] ,task_config["tile_size"][1])

        #Calculate the centroids for the slices along x and y for the ZI region.
        x_zi = np.arange(task_config["tile_size"][0]/2, task_config["xrange_zi"][1]-task_config["xrange_zi"][0] ,task_config["tile_size"][0])
        y_zi = np.arange(task_config["tile_size"][1]/2, task_config["yrange_zi"][1]-task_config["yrange_zi"][0] ,task_config["tile_size"][1])


        #Fetch the z ranges for train test and valiation sets (they are stacked on top of each other along the Z direction).
        if mode == "train":
            z_vals = task_config["z_train"]
        elif mode == "val":
            z_vals = task_config["z_val"]
        elif mode == "test":
            z_vals = task_config["z_test"]
            
        #If specified to use pre-downloaded data and if it exists, use it.
        if download and exists(download_path+task_config['name']+mode+'images.npy'):
            image_array = np.load(download_path+task_config['name']+mode+'images.npy')
            mask_array = np.load(download_path+task_config['name']+mode+'labels.npy')
            self.image_array = image_array
            self.mask_array = mask_array
        #If not specified to use pre-downloaded data or if pre-downloaded data doesn't exist.
        else:
            self.config = boss_config

            reset_counter = 0
            
            #Connect to BossDB using array method from intern library and download data, retry if it fails.
            while reset_counter<retries:
                try:
                    print('Downloading BossDB cutout...')
                    self.boss_image_array = array(task_config["image_chan"], boss_config=boss_config)
                    self.boss_mask_array = array(task_config["annotation_chan"], boss_config=boss_config)
                    #Use the x, y and z ranges to select the appropriate set of slices and their annotations for each region.
                    #cortex
                    image_array =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_cor"][0] : task_config["yrange_cor"][1],
                            task_config["xrange_cor"][0] : task_config["xrange_cor"][1],
                        ]
                    mask_array =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_cor"][0] : task_config["yrange_cor"][1],
                            task_config["xrange_cor"][0] : task_config["xrange_cor"][1],
                        ]
                    #striatum
                    image_array_temp =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_stri"][0] : task_config["yrange_stri"][1],
                            task_config["xrange_stri"][0] : task_config["xrange_stri"][1],
                        ]
                    mask_array_temp =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_stri"][0] : task_config["yrange_stri"][1],
                            task_config["xrange_stri"][0] : task_config["xrange_stri"][1],
                        ]
                    #concatenate the striatum slices on top of cortex slices.
                    image_array = np.concatenate((image_array,image_array_temp))
                    mask_array = np.concatenate((mask_array,mask_array_temp))
                    #vp
                    image_array_temp =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_vp"][0] : task_config["yrange_vp"][1],
                            task_config["xrange_vp"][0] : task_config["xrange_vp"][1],
                        ]
                    mask_array_temp =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_vp"][0] : task_config["yrange_vp"][1],
                            task_config["xrange_vp"][0] : task_config["xrange_vp"][1],
                        ]
                    #concatenate the VP slices on top of the previous set ending with striatum. 
                    image_array = np.concatenate((image_array,image_array_temp))
                    mask_array = np.concatenate((mask_array,mask_array_temp))
                    #zi
                    image_array_temp =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_zi"][0] : task_config["yrange_zi"][1],
                            task_config["xrange_zi"][0] : task_config["xrange_zi"][1],
                        ]
                    mask_array_temp =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_zi"][0] : task_config["yrange_zi"][1],
                            task_config["xrange_zi"][0] : task_config["xrange_zi"][1],
                        ]
                    #concatenate the ZI slices on top of the previous set ending with VP.
                    image_array = np.concatenate((image_array,image_array_temp))
                    mask_array = np.concatenate((mask_array,mask_array_temp))
                    self.image_array = image_array
                    self.mask_array = mask_array
                    #If 'download' is specified, save the downloaded data as '.npy' file.
                    if download:
                        np.save(download_path+task_config['name']+mode+'images.npy', image_array)
                        np.save(download_path+task_config['name']+mode+'labels.npy', mask_array)
                    break
                #In case the connection to BossDB fails
                except HTTPError as e:
                    print('Error connecting to BossDB channels, retrying')
                    print(e)
                    reset_counter = reset_counter + 1

        #note- for X and Y, this is the centroid, for the z dimension the cutout is handled differently
        #z is the start of the volume, and the volume extends to z+task_config["volume_z"]. Size of a volume -> task_config["volume_z"]
        #Each region is stacked on top of each other along Z in the order -> Cortex, Striatum, VP, ZI.
        centroids = []
        #collect all x and y centroids for train set.
        if mode == "train":
            #cortex
            z_vals = np.arange(0,task_config["z_train"][1]-task_config["z_train"][0] ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            
            #striatum
            z_vals = np.arange((task_config["z_train"][1]-task_config["z_train"][0]),2*(task_config["z_train"][1]-task_config["z_train"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_stri:
                    for y in y_stri:
                        centroids.append([z, y, x])

            #vp
            z_vals = np.arange(2*(task_config["z_train"][1]-task_config["z_train"][0]),3*(task_config["z_train"][1]-task_config["z_train"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_vp:
                    for y in y_vp:
                        centroids.append([z, y, x])

            #If it specified as 4 class setting in task config then dont collect the centroids for ZI.
            if 'noZI' in task_config and bool(task_config['noZI']):
                pass
            else:
                #zi
                z_vals = np.arange(3*(task_config["z_train"][1]-task_config["z_train"][0]),4*(task_config["z_train"][1]-task_config["z_train"][0]) ,task_config["volume_z"])
            
                for z in z_vals:
                    for x in x_zi:
                        for y in y_zi:
                            centroids.append([z, y, x])
        
        #Collect all x and y centroids for the val set.
        if mode == "val":
            z_vals = np.arange(0,task_config["z_val"][1]-task_config["z_val"][0] ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            z_vals = np.arange((task_config["z_val"][1]-task_config["z_val"][0]),2*(task_config["z_val"][1]-task_config["z_val"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            z_vals = np.arange(2*(task_config["z_val"][1]-task_config["z_val"][0]),3*(task_config["z_val"][1]-task_config["z_val"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
             
            #If it specified as 4 class setting in the task config then dont collect the centroids for ZI.
            if 'noZI' in task_config and bool(task_config['noZI']):
                pass
            else:
                z_vals = np.arange(3*(task_config["z_val"][1]-task_config["z_val"][0]),4*(task_config["z_val"][1]-task_config["z_val"][0]) ,task_config["volume_z"])
                for z in z_vals:
                    for x in x_cor:
                        for y in y_cor:
                            centroids.append([z, y, x])

        if mode == "test":
            z_vals = np.arange(0,task_config["z_test"][1]-task_config["z_test"][0] ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            
            z_vals = np.arange((task_config["z_test"][1]-task_config["z_test"][0]),2*(task_config["z_test"][1]-task_config["z_test"][0]) ,task_config["volume_z"])         
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            z_vals = np.arange(2*(task_config["z_test"][1]-task_config["z_test"][0]),3*(task_config["z_test"][1]-task_config["z_test"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            #If it specified as 4 class setting in the task config then dont collect the centroids for ZI.             
            if 'noZI' in task_config and bool(task_config['noZI']):
                pass
            else:
                z_vals = np.arange(3*(task_config["z_test"][1]-task_config["z_test"][0]),4*(task_config["z_test"][1]-task_config["z_test"][0]) ,task_config["volume_z"])
                
                for z in z_vals:
                    for x in x_cor:
                        for y in y_cor:
                            centroids.append([z, y, x])
        #List of slices
        self.centroid_list = centroids
        rad_y = int(task_config["tile_size"][1]/2)
        rad_x = int(task_config["tile_size"][0]/2)
        self.px_radius_y = rad_y
        self.px_radius_x = rad_x

        # Deprecate mask_transform
        # test if transform = transforms.ToTensor()
        if isinstance(mask_transform, transforms.ToTensor):
            mask_transform = None
        # test if transform = transforms.Compose([transforms.ToTensor(),])
        elif isinstance(mask_transform, transforms.Compose) and len(mask_transform.transforms) == 1 and isinstance(
            mask_transform.transforms[0], transforms.ToTensor):
            mask_transform = None
        elif mask_transform is not None:
            raise DeprecationWarning('mask_transform is deprecated, use transform.')

        # image_transform
        if isinstance(image_transform, transforms.ToTensor):
            image_transform = None
        # test if transform = transforms.Compose([transforms.ToTensor(),])
        elif isinstance(image_transform, transforms.Compose) and len(image_transform.transforms) == 1 and isinstance(
            image_transform.transforms[0], transforms.ToTensor):
            image_transform = None
        elif image_transform is not None:
            warnings.warn('image_transform does not require transforms.ToTensor().')

        self.transform = transform
        self.image_transform = image_transform

        self.z_size = task_config["volume_z"]
        if 'combine_ax_and_bg' in task_config and bool(task_config['combine_ax_and_bg']):
            self.combine_ax_and_bg = 1
        else:
            self.combine_ax_and_bg = 0
        #checking if the task specified is task1.
        self.task1 = True if task_config["name"] == 'task1' else False

    def _get_img_label(self, mask):
        """
        Processes the annotations for Task 1.

        Args: 
            mask:  The orginial annotations.
        Returns:
            label: Image labels processed for Task 1.
        """
        roi_label = torch.mode(mask.flatten())[0]
        if roi_label == 1 or roi_label == 0:
            img_label = 0
        elif roi_label == 2 or roi_label == 8:
            img_label = 1
        elif roi_label == 3 or roi_label == 4:
            img_label = 2
        elif roi_label == 5 or roi_label == 6 or roi_label == 7:
            img_label = 3
        return img_label

    def __getitem__(self, key):
        """
        Get the a specific slice/volume by index.

        Args:
            key:                    Slice/volume number to pick from the set.
        Return:
            image_array:            The slice/volume that was specified.
            mask_array:             The annotation corresponding to the slice/volume specified.
            threeclass_mask_array:  The annotation corresponding to the slice/volume specified, for the 3-class setting.
        """
        z, y, x = self.centroid_list[key]
        z = int(z)
        y = int(y)
        x = int(x)

        #Pick the slice/volume and annotation corresponding specified from the data.
        image_array =  self.image_array[
                z : z + self.z_size,
                y - self.px_radius_y : y + self.px_radius_y,
                x - self.px_radius_x : x + self.px_radius_x,
            ]
        mask_array =  self.mask_array[
                z : z + self.z_size,
                y - self.px_radius_y : y + self.px_radius_y,
                x - self.px_radius_x : x + self.px_radius_x,
            ]
       
        #If some transform has been specified
        # Transpose to (c, z, x, y) and add channel dimension
        image_array = np.transpose(image_array, (0, 2, 1))[np.newaxis, :]
        mask_array = np.transpose(mask_array, (0, 2, 1))[np.newaxis, :]

        if self.z_size == 1:
            # Squeeze z-axis for shape (c, x, y)
            image_array = np.squeeze(image_array, axis=1)
            mask_array = np.squeeze(mask_array, axis=0)
        image = torch.FloatTensor(image_array) / 255.
        mask = torch.FloatTensor(mask_array.astype(np.int64))

        # apply transform
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # apply image_transform
        if self.image_transform is not None:
            image = self.image_transform(image)

        # convert mask to long
        # todo make it such that by default  of type float
        mask = mask.long().squeeze(0)

        if self.task1:
            y = self._get_img_label(mask)
            return image, y

        # If it is set in the task config to combine the axons and the background labels then do it.
        if self.combine_ax_and_bg:
            # map class 3 to 0
            mask = torch.LongTensor([0, 1, 2, 0])[mask]

        return image, mask

    def __len__(self):
        """
            Number of slices in the downloaded data can be checked by applying len() method.

        Returns:
            int: Number of slices
        """
        return len(self.centroid_list)
