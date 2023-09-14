import os
import boto3
import numpy as np

# deal with paths where we want the output
base_dir =  os.path.join('/home', 'charestlab')
to_nsd_dir = os.path.join(base_dir, 'data', 'natural-scenes-dataset')

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
    this_file = file.key
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
    inclusion = ['fsaverage', 'betas_fithrf_GLMdenoise_RR', 'mgh', 'betas_session', sub]

    for file in nsd.objects.filter(Prefix="nsddata_betas/ppdata"):
        if np.all([x in file.key for x in inclusion]) and not np.all([x in file.key for x in exclusion]):
            this_file = file.key
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
            this_file = file.key
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