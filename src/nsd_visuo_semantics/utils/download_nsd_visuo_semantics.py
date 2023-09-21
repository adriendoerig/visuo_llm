import os
import boto3
import numpy as np

# deal with paths where we want the output
to_nsd_dir = os.path.join('/share/klab/datasets/', 'NSD_for_visuo_semantics')

# we are downloading the natural scenes dataset
bucket = 'natural-scenes-dataset'

# let's open a boto3 resource
s3 = boto3.resource('s3', region_name='us-east-2')
nsd = s3.Bucket(bucket)

# and a client for downloading files
nsdc = boto3.client('s3', region_name='us-east-2')

# download the nsddata dir    # this is required for the transform files
# to work later with nsdcode
c = 0
size = 0
for file in nsd.objects.filter(Prefix="nsddata/"):
    print(f'Downloading {file.key}')
    # deal with directory hierarchy

    # deal with directory hierarchy
    aws_file_dir = os.path.split(file.key)[0]

    # join the aws base dir to our nsd target dir
    target_dir = os.path.join(to_nsd_dir, aws_file_dir)

    # make sure the destination exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # target path and filename
    target_file = os.path.join(
        target_dir,
        os.path.basename(file.key)
    )

    # let's move it (but not the imagery data, which we don't have access to)!
    if 'nsdimagery' not in target_file and 'nsdsynthetic' not in target_file:
        nsdc.download_file(bucket, file.key, target_file) 

    c+=1
    size += file.size

print('Total size of nsddata in GB: ' + str(size / (1000**3)))


# download the fmri data # that part you can skip unless you want to dl it yourself
# we will list some files, and then include / exclude some files.
exclusion = ['meanbeta']
c = 0
size = 0

subjects = [f'subj0{x}' for x in range(1,9)]
for sub in subjects:
    inclusion = ['fsaverage', 'betas_fithrf_GLMdenoise_RR', 'mgh', 'betas_session', sub]

    for file in nsd.objects.filter(Prefix="nsddata_betas/ppdata"):
        if np.all([x in file.key for x in inclusion]) and not np.all([x in file.key for x in exclusion]):
            print(f'Downloading {file.key}')
            # deal with directory hierarchy
            # check if we can dl it
            r = nsdc.list_objects_v2(Bucket=bucket, Prefix=file.key)

            # deal with directory hierarchy
            aws_file_dir = os.path.split(file.key)

            # join the aws base dir to our nsd target dir
            target_dir = os.path.join(to_nsd_dir, aws_file_dir[0])

            # make sure the destination exists
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # target path and filename
            target_file = os.path.join(
                target_dir,
                aws_file_dir[1]
            )

            # let's move it!
            nsdc.download_file(bucket, file.key, target_file)

            c+=1
            size += file.size
print('Total size of nsddata in GB: ' + str(size / (1000**3)))

# download the fmri data # that part you can skip unless you want to dl it yourself
# we will list some files, and then include / exclude some files.
exclusion = ['meanbeta']
c = 0
size = 0
# subjects = ['subj01']
subjects = [f'subj0{x}' for x in range(1,9)]
for sub in subjects:
    inclusion = ['func1pt8mm', 'betas_fithrf_GLMdenoise_RR', 'nii.gz', 'betas_session', sub]

    for file in nsd.objects.filter(Prefix="nsddata_betas/ppdata"):
        if np.all([x in file.key for x in inclusion]) and not np.all([x in file.key for x in exclusion]):
            print(f'Downloading {file.key}')
            # deal with directory hierarchy
            # check if we can dl it
            r = nsdc.list_objects_v2(Bucket=bucket, Prefix=file.key)

            # deal with directory hierarchy
            aws_file_dir = os.path.split(file.key)

            # join the aws base dir to our nsd target dir
            target_dir = os.path.join(to_nsd_dir, aws_file_dir[0])

            # make sure the destination exists
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            # target path and filename
            target_file = os.path.join(
                target_dir,
                aws_file_dir[1]
            )

            # let's move it!
            nsdc.download_file(bucket, file.key, target_file)

            c+=1
            size += file.size

nsd_stimuli_dir = os.path.join(to_nsd_dir, 'nsddata_stimuli/stimuli/nsd')
os.makedirs(nsd_stimuli_dir, exist_ok=True)
file = [x for x in nsd.objects.filter(Prefix='nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')][0]
nsdc.download_file(bucket, file.key, nsd_stimuli_dir+'/nsd_stimuli.hdf5')
c+=1
size += file.size

from nsd_access import NSDAccess
nsda = NSDAccess(to_nsd_dir)
nsda.download_coco_annotation_file()

print('Total size of nsddata in GB: ' + str(size / (1000**3)))