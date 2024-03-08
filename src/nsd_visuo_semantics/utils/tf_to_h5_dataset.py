import tensorflow_datasets as tfds
import h5py, os
import numpy as np

CREATE_FULL_DATASET = 1
CREATE_TRAIN_SUBSET = 0
CHECK_DATASET = 1
DEBUG = False

tf_dir = "/share/klab/datasets"
h5_path = f"/share/klab/datasets/places365_small"
check_imgs_path = f"/share/klab/datasets/places365_small/check_imgs"

h5_dtype = {'data': 'uint8', 'labels': 'int32'}
tf_to_h5_keys = {'image': 'data', 'label': 'labels'}
h5_to_tf_keys = {v: k for k, v in tf_to_h5_keys.items()}

tf_keys = ['image', 'label']
tf_shapes = {}
tf_dtype = {}

# Load the dataset (tf_data[0] = train, tf_data[1] = validation, tf_data[2] = test)
tf_data = tfds.load('places365_small', split=['train', 'validation'], data_dir=tf_dir)  # we don't do test because the labels are not available
if CREATE_TRAIN_SUBSET:
    assert not CREATE_FULL_DATASET, 'Cannot create both full dataset and train subset'
    tf_data = [tf_data[0].take(73000), tf_data[1]]
    dataset_name = 'places365_small_first73000train.h5'
else:
    dataset_name = 'places365_small.h5'

if DEBUG:
    tf_data = [tf_data[0].take(10000), tf_data[1].take(100)]
    dataset_name = 'places365_small_debug.h5'

tf_lengths = {'train': len(tf_data[0]), 'validation': len(tf_data[1])}   # we don't do test because the labels are not available


tf_dummy = tf_data[0].take(1)
for i in tf_dummy:
    for k in tf_keys:
        tf_shapes[k] = i[k].shape
        tf_dtype[k] = i[k].dtype

label2cat = {
        0: 'airfield',
        1: 'airplane_cabin',
        2: 'airport_terminal',
        3: 'alcove',
        4: 'alley',
        5: 'amphitheater',
        6: 'amusement_arcade',
        7: 'amusement_park',
        8: 'apartment_building/outdoor',
        9: 'aquarium',
        10: 'aqueduct',
        11: 'arcade',
        12: 'arch',
        13: 'archaelogical_excavation',
        14: 'archive',
        15: 'arena/hockey',
        16: 'arena/performance',
        17: 'arena/rodeo',
        18: 'army_base',
        19: 'art_gallery',
        20: 'art_school',
        21: 'art_studio',
        22: 'artists_loft',
        23: 'assembly_line',
        24: 'athletic_field/outdoor',
        25: 'atrium/public',
        26: 'attic',
        27: 'auditorium',
        28: 'auto_factory',
        29: 'auto_showroom',
        30: 'badlands',
        31: 'bakery/shop',
        32: 'balcony/exterior',
        33: 'balcony/interior',
        34: 'ball_pit',
        35: 'ballroom',
        36: 'bamboo_forest',
        37: 'bank_vault',
        38: 'banquet_hall',
        39: 'bar',
        40: 'barn',
        41: 'barndoor',
        42: 'baseball_field',
        43: 'basement',
        44: 'basketball_court/indoor',
        45: 'bathroom',
        46: 'bazaar/indoor',
        47: 'bazaar/outdoor',
        48: 'beach',
        49: 'beach_house',
        50: 'beauty_salon',
        51: 'bedchamber',
        52: 'bedroom',
        53: 'beer_garden',
        54: 'beer_hall',
        55: 'berth',
        56: 'biology_laboratory',
        57: 'boardwalk',
        58: 'boat_deck',
        59: 'boathouse',
        60: 'bookstore',
        61: 'booth/indoor',
        62: 'botanical_garden',
        63: 'bow_window/indoor',
        64: 'bowling_alley',
        65: 'boxing_ring',
        66: 'bridge',
        67: 'building_facade',
        68: 'bullring',
        69: 'burial_chamber',
        70: 'bus_interior',
        71: 'bus_station/indoor',
        72: 'butchers_shop',
        73: 'butte',
        74: 'cabin/outdoor',
        75: 'cafeteria',
        76: 'campsite',
        77: 'campus',
        78: 'canal/natural',
        79: 'canal/urban',
        80: 'candy_store',
        81: 'canyon',
        82: 'car_interior',
        83: 'carrousel',
        84: 'castle',
        85: 'catacomb',
        86: 'cemetery',
        87: 'chalet',
        88: 'chemistry_lab',
        89: 'childs_room',
        90: 'church/indoor',
        91: 'church/outdoor',
        92: 'classroom',
        93: 'clean_room',
        94: 'cliff',
        95: 'closet',
        96: 'clothing_store',
        97: 'coast',
        98: 'cockpit',
        99: 'coffee_shop',
        100: 'computer_room',
        101: 'conference_center',
        102: 'conference_room',
        103: 'construction_site',
        104: 'corn_field',
        105: 'corral',
        106: 'corridor',
        107: 'cottage',
        108: 'courthouse',
        109: 'courtyard',
        110: 'creek',
        111: 'crevasse',
        112: 'crosswalk',
        113: 'dam',
        114: 'delicatessen',
        115: 'department_store',
        116: 'desert/sand',
        117: 'desert/vegetation',
        118: 'desert_road',
        119: 'diner/outdoor',
        120: 'dining_hall',
        121: 'dining_room',
        122: 'discotheque',
        123: 'doorway/outdoor',
        124: 'dorm_room',
        125: 'downtown',
        126: 'dressing_room',
        127: 'driveway',
        128: 'drugstore',
        129: 'elevator/door',
        130: 'elevator_lobby',
        131: 'elevator_shaft',
        132: 'embassy',
        133: 'engine_room',
        134: 'entrance_hall',
        135: 'escalator/indoor',
        136: 'excavation',
        137: 'fabric_store',
        138: 'farm',
        139: 'fastfood_restaurant',
        140: 'field/cultivated',
        141: 'field/wild',
        142: 'field_road',
        143: 'fire_escape',
        144: 'fire_station',
        145: 'fishpond',
        146: 'flea_market/indoor',
        147: 'florist_shop/indoor',
        148: 'food_court',
        149: 'football_field',
        150: 'forest/broadleaf',
        151: 'forest_path',
        152: 'forest_road',
        153: 'formal_garden',
        154: 'fountain',
        155: 'galley',
        156: 'garage/indoor',
        157: 'garage/outdoor',
        158: 'gas_station',
        159: 'gazebo/exterior',
        160: 'general_store/indoor',
        161: 'general_store/outdoor',
        162: 'gift_shop',
        163: 'glacier',
        164: 'golf_course',
        165: 'greenhouse/indoor',
        166: 'greenhouse/outdoor',
        167: 'grotto',
        168: 'gymnasium/indoor',
        169: 'hangar/indoor',
        170: 'hangar/outdoor',
        171: 'harbor',
        172: 'hardware_store',
        173: 'hayfield',
        174: 'heliport',
        175: 'highway',
        176: 'home_office',
        177: 'home_theater',
        178: 'hospital',
        179: 'hospital_room',
        180: 'hot_spring',
        181: 'hotel/outdoor',
        182: 'hotel_room',
        183: 'house',
        184: 'hunting_lodge/outdoor',
        185: 'ice_cream_parlor',
        186: 'ice_floe',
        187: 'ice_shelf',
        188: 'ice_skating_rink/indoor',
        189: 'ice_skating_rink/outdoor',
        190: 'iceberg',
        191: 'igloo',
        192: 'industrial_area',
        193: 'inn/outdoor',
        194: 'islet',
        195: 'jacuzzi/indoor',
        196: 'jail_cell',
        197: 'japanese_garden',
        198: 'jewelry_shop',
        199: 'junkyard',
        200: 'kasbah',
        201: 'kennel/outdoor',
        202: 'kindergarden_classroom',
        203: 'kitchen',
        204: 'lagoon',
        205: 'lake/natural',
        206: 'landfill',
        207: 'landing_deck',
        208: 'laundromat',
        209: 'lawn',
        210: 'lecture_room',
        211: 'legislative_chamber',
        212: 'library/indoor',
        213: 'library/outdoor',
        214: 'lighthouse',
        215: 'living_room',
        216: 'loading_dock',
        217: 'lobby',
        218: 'lock_chamber',
        219: 'locker_room',
        220: 'mansion',
        221: 'manufactured_home',
        222: 'market/indoor',
        223: 'market/outdoor',
        224: 'marsh',
        225: 'martial_arts_gym',
        226: 'mausoleum',
        227: 'medina',
        228: 'mezzanine',
        229: 'moat/water',
        230: 'mosque/outdoor',
        231: 'motel',
        232: 'mountain',
        233: 'mountain_path',
        234: 'mountain_snowy',
        235: 'movie_theater/indoor',
        236: 'museum/indoor',
        237: 'museum/outdoor',
        238: 'music_studio',
        239: 'natural_history_museum',
        240: 'nursery',
        241: 'nursing_home',
        242: 'oast_house',
        243: 'ocean',
        244: 'office',
        245: 'office_building',
        246: 'office_cubicles',
        247: 'oilrig',
        248: 'operating_room',
        249: 'orchard',
        250: 'orchestra_pit',
        251: 'pagoda',
        252: 'palace',
        253: 'pantry',
        254: 'park',
        255: 'parking_garage/indoor',
        256: 'parking_garage/outdoor',
        257: 'parking_lot',
        258: 'pasture',
        259: 'patio',
        260: 'pavilion',
        261: 'pet_shop',
        262: 'pharmacy',
        263: 'phone_booth',
        264: 'physics_laboratory',
        265: 'picnic_area',
        266: 'pier',
        267: 'pizzeria',
        268: 'playground',
        269: 'playroom',
        270: 'plaza',
        271: 'pond',
        272: 'porch',
        273: 'promenade',
        274: 'pub/indoor',
        275: 'racecourse',
        276: 'raceway',
        277: 'raft',
        278: 'railroad_track',
        279: 'rainforest',
        280: 'reception',
        281: 'recreation_room',
        282: 'repair_shop',
        283: 'residential_neighborhood',
        284: 'restaurant',
        285: 'restaurant_kitchen',
        286: 'restaurant_patio',
        287: 'rice_paddy',
        288: 'river',
        289: 'rock_arch',
        290: 'roof_garden',
        291: 'rope_bridge',
        292: 'ruin',
        293: 'runway',
        294: 'sandbox',
        295: 'sauna',
        296: 'schoolhouse',
        297: 'science_museum',
        298: 'server_room',
        299: 'shed',
        300: 'shoe_shop',
        301: 'shopfront',
        302: 'shopping_mall/indoor',
        303: 'shower',
        304: 'ski_resort',
        305: 'ski_slope',
        306: 'sky',
        307: 'skyscraper',
        308: 'slum',
        309: 'snowfield',
        310: 'soccer_field',
        311: 'stable',
        312: 'stadium/baseball',
        313: 'stadium/football',
        314: 'stadium/soccer',
        315: 'stage/indoor',
        316: 'stage/outdoor',
        317: 'staircase',
        318: 'storage_room',
        319: 'street',
        320: 'subway_station/platform',
        321: 'supermarket',
        322: 'sushi_bar',
        323: 'swamp',
        324: 'swimming_hole',
        325: 'swimming_pool/indoor',
        326: 'swimming_pool/outdoor',
        327: 'synagogue/outdoor',
        328: 'television_room',
        329: 'television_studio',
        330: 'temple/asia',
        331: 'throne_room',
        332: 'ticket_booth',
        333: 'topiary_garden',
        334: 'tower',
        335: 'toyshop',
        336: 'train_interior',
        337: 'train_station/platform',
        338: 'tree_farm',
        339: 'tree_house',
        340: 'trench',
        341: 'tundra',
        342: 'underwater/ocean_deep',
        343: 'utility_room',
        344: 'valley',
        345: 'vegetable_garden',
        346: 'veterinarians_office',
        347: 'viaduct',
        348: 'village',
        349: 'vineyard',
        350: 'volcano',
        351: 'volleyball_court/outdoor',
        352: 'waiting_room',
        353: 'water_park',
        354: 'water_tower',
        355: 'waterfall',
        356: 'watering_hole',
        357: 'wave',
        358: 'wet_bar',
        359: 'wheat_field',
        360: 'wind_farm',
        361: 'windmill',
        362: 'yard',
        363: 'youth_hostel',
        364: 'zen_garden'
    }

categories = list(label2cat.values())

if CREATE_FULL_DATASET or CREATE_TRAIN_SUBSET:

    with h5py.File(f'{h5_path}/{dataset_name}', "w") as f:

        f.create_dataset('categories', data=categories, dtype='S100')

        for s, split in enumerate(['train', 'validation']):  # we don't do test because the labels are not available

            print('Creating group:', split)
            h5_split = 'val' if split == 'validation' else split
            grp = f.create_group(h5_split)

            for k in tf_keys:

                print(f'Creating dataset:{tf_to_h5_keys[k]} with shape: {(tf_lengths[split],)+tf_shapes[k]}, dtype: {h5_dtype[tf_to_h5_keys[k]]}, and chunks: {tuple((1,)+tf_shapes[k])}')

                grp.create_dataset(tf_to_h5_keys[k], 
                                data=np.empty((tf_lengths[split],)+tf_shapes[k], dtype=h5_dtype[tf_to_h5_keys[k]]), 
                                chunks=tuple((1,)+tf_shapes[k]),
                                dtype=h5_dtype[tf_to_h5_keys[k]])

            for it, this_item in enumerate(tf_data[s]):

                if it % 1000 == 0:
                    print(f'Processing item {it} of {tf_lengths[split]}', end='\r')
                
                grp[tf_to_h5_keys['image']][it] = this_item['image'].numpy()
                grp[tf_to_h5_keys['label']][it] = this_item['label'].numpy()


# if CREATE_TRAIN_SUBSET:

    # with h5py.File(f'{h5_path}/{dataset_name}', "r") as f:

    #     with h5py.File(f'{h5_path}/{dataset_name}'.replace('.h5', '_first73000train.h5'), "w") as f_subset:

    #         f.create_dataset('categories', data=categories, dtype='S100')

    #         for s, split in enumerate(['train']):

    #             print('Creating group:', split)
    #             grp = f_subset.create_group(split)

    #             for k in tf_keys:
                    
    #                 print(f'Creating dataset:{tf_to_h5_keys[k]} with shape: {f[split][tf_to_h5_keys[k]][:73000].shape}, dtype: {h5_dtype[tf_to_h5_keys[k]]}, and chunks: {tuple([1]+tf_shapes[k])}')
                    
    #                 grp.create_dataset(tf_to_h5_keys[k], 
    #                                 data=f[split][tf_to_h5_keys[k]][:73000], 
    #                                 chunks=tuple([1]+tf_shapes[k]),
    #                                 dtype=h5_dtype[tf_to_h5_keys[k]])

if CHECK_DATASET:

    os.makedirs(check_imgs_path, exist_ok=True)

    # plot a few images from each group
    import matplotlib.pyplot as plt
    with h5py.File(f'{h5_path}/{dataset_name}', "r") as f:
        for s, split in enumerate(['train', 'validation']):

            print('Plotting group:', split)
            h5_split = 'val' if split == 'validation' else split
            grp = f[h5_split]
            print(f'Plotting dataset:{tf_to_h5_keys[k]} with shape: {(tf_lengths[split],)+tf_shapes[k]}, dtype: {h5_dtype[tf_to_h5_keys[k]]}')
            for i in np.random.choice(range(tf_lengths[split]), 5):
                plt.imshow(grp['data'][i])
                label = grp['labels'][i]
                plt.title(f'Label: {label}: {f["categories"][label]}')
                plt.savefig(os.path.join(check_imgs_path, f'{split}_{i}.png'))
                plt.close()
                print(f'Plotting image {i} of {tf_lengths[split]}', end='\r')

# if CHECK_DATASET_SUBSET:

#     os.makedirs(check_imgs_path, exist_ok=True)

#     # plot a few images from each group
#     import matplotlib.pyplot as plt
#     with h5py.File(f'{h5_path}/{dataset_name}'.replace('.h5', '_first73000train.h5'), "r") as f:
#         for s, split in enumerate(['train']):
#             h5_split = 'val' if split == 'validation' else split
#             print('Plotting group:', split)
#             print('Data shape:', f[split]['data'].shape)
#             print('Labels shape:', f[split]['labels'].shape)
#             grp = f[h5_split]
#             print(f'Plotting dataset:{tf_to_h5_keys[k]} with shape: {(tf_lengths[split],)+tf_shapes[k]}, dtype: {h5_dtype[tf_to_h5_keys[k]]}')
#             for i in np.random.choice(range(73000), 5):
#                 plt.imshow(grp['data'][i])
#                 plt.title(f'Label: {grp["labels"][i]}: {label2cat[grp["labels"][i]]}')
#                 plt.savefig(os.path.join(check_imgs_path, f'train73000subset_{split}_{i}.png'))
#                 plt.close()
#                 print(f'Plotting image {i} of {73000}', end='\r')