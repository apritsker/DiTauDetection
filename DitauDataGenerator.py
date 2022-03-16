"""
DitauDataGenerator.py:
Handles processing the raw csv files and turn them into a sample-based dataset.
"""
import os
from collections import defaultdict
import pickle
import glob
import tarfile
from math import log, sqrt, pi

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import DataVisualization

def GenerateDitauData(input_signal_path, input_bg_path, out_dir, output_tar_path, image_size, generate_tar=True, retain_extracted=True, equal_num_background_to_samples = False):
    """
    The main function of this file.
    This function processes 2 files: signal and background file, splits them into separate sample files, and generates a tar of the output directory.
    :param input_signal_path: csv file of the recorded signal
    :param input_bg_path: csv file of the recorded background signals
    :param output_tar_path: the path to the output tar file
    :param tar_only: Default is True - to remove intermediate files
    :return: None
    """
    clean = FolderCleanup(out_dir)
    if not clean:
        print("Could not clean out dir. Aborting.")
        return
    
    cur_sample_num = 0

    signal_linecount = _GetLinecount(input_signal_path)
    background_linecount = _GetLinecount(input_bg_path)

    print("\nExtracting normalization data...")
    outfile_path = out_dir + 'normalization_data.pkl'
    norm_data = _ExtractNormalizationData_Running(input_signal_path, signal_linecount, input_bg_path, background_linecount)
    with open(outfile_path, 'wb') as fo:
        pickle.dump(norm_data, fo)
    print("\nDone. Saved normalization data to " + outfile_path)


    print("\nProcessing signal file: " + input_signal_path + "...")
    #cur_sample_num = _SplitCSVToSampleFiles(input_signal_path, signal_linecount, 1, norm_data, image_size, cur_sample_num, out_dir)
    num_signal_samples = cur_sample_num + 1
    print("\nGenerated " + str(num_signal_samples) + " Signal files.")
    print("\nProcessing background file: " + input_bg_path + "...")
    if equal_num_background_to_samples:
        cur_sample_num = _SplitCSVToSampleFiles(input_bg_path, background_linecount, 0, norm_data, image_size, cur_sample_num, out_dir, cur_sample_num)
    else:
        cur_sample_num = _SplitCSVToSampleFiles(input_bg_path, background_linecount, 0, norm_data, image_size, cur_sample_num, out_dir)
    num_background_samples = cur_sample_num + 1 - num_signal_samples
    print("\nGenerated " + str(num_background_samples) + " Background files.")


    outfile_path = out_dir + '/track_sizes.pkl'
    print("Generating tracks sizes file...")
    _GenerateTrackSizesFile(outfile_path=outfile_path, source_dir=out_dir)   


    if generate_tar:        
        print("\nGenerating tar file...")
        MakeTarfile(output_tar_path, out_dir)


    print("\nComplete!")
    if not retain_extracted:
        print("\n Cleaning Cache...")
        FolderCleanup(out_dir)





def FolderCleanup(dir_path):
    userin = input("You are about to clean the " + dir_path + " folder. Are you sure? y/n ")
    if(userin[0].lower() == 'y'):
        print("\nCleaning up folder: " + dir_path + "...")    
        del_files = glob.glob(dir_path + '/*')
        if not del_files:
            print("\nComplete.")
            return True
        for f in tqdm(del_files):
            os.remove(f)
        print("\nComplete.")
        return True

    return False


def MakeTarfile(outfile_path, source_dir):
    print("\nPackaging processed dataset to tar file: " + outfile_path + "...")
    with tarfile.open(outfile_path, "w:gz") as tar:
        tar.add(source_dir, arcname='')
        # tar.add(source_dir, arcname=os.path.basename(source_dir))



def _GenerateTrackSizesFile(outfile_path, source_dir):
    print("\nGenerating track sizes file...")
    num_files = len(glob.glob(source_dir + '/*')) - 2
    track_sizes = np.zeros((num_files,))
    for i in tqdm(range(num_files)):
        file_path = source_dir + '/sample_' + str(i).zfill(6) + '.pkl'
        with open(file_path, 'rb') as fi:
            obj = pickle.load(fi)
            track_sizes[i] = obj['num_tracks']

    with open(outfile_path, 'wb') as fo:
        pickle.dump(track_sizes, fo)
    print("\nDone. Saved track sizes data to " + outfile_path)



def _SplitCSVToSampleFiles(csv_path, linecount, label, normalization_data, image_size, global_sample_num, out_dir):
    """
        This function processes the csv file efficiently assuming the samples in the file are already pre-grouped
            so there's no need to load everything to memory and sort it.
            The function scans line by line and adds to the sample, and when it detects a new sample it packages the previous one to a separate file.
        :param csv_path: path to csv file.
        :param label: the label to be attached to every sample in the file. '1' for signal, '0' for background.
            This function assumes all samples in a csv file belong to the same label.
        :param global_sample_num: a counter for the total samples created so far. Used for naming the created sample files.
            Useful when using this function with multiple csv files consecutively.
        :param out_dir: path to output directory to be populate with sample files.
        :return: sample_num: the current last sample number.
            populates output directory with separate sample files.
    """
    desired_num = None
    with open(csv_path, "r") as f:
        header_line = next(f)
        categories = ['event_num', 'sample', 'signal', 'eta', 'phi', 'EM', 'Had', 'Largeeta', 'Largephi']

        # convert each csv line into a dict, and append all into a list of dicts.
        print("\nProcessing lines...")

        current_event_num = '-1'
        current_sample_num = '-1'

        line_dicts_em = []
        line_dicts_had = []
        line_dicts_tracks = []

        generated_samples = 0

        for line in tqdm(f, total=linecount):

            # turn the line into a dictionary with key-value pairs.
            line_split = line.strip('\n').split(',')
            line_dict = defaultdict()
            i = 0
            for entry in line_split:
                line_dict[categories[i]] = entry
                i += 1

            # if this is a new sample: package the previous sample to a separate file and create a new collection for the new sample.
            if line_dict['event_num'] != current_event_num or line_dict['sample'] != current_sample_num:
                if current_event_num != '-1' and line_dicts_em and line_dicts_had and line_dicts_tracks:
                    _PackageSample(line_dicts_em, line_dicts_had, line_dicts_tracks, image_size, label, normalization_data, global_sample_num, out_dir)
                    global_sample_num += 1
                    generated_samples += 1
                    if desired_num != None and generated_samples > desired_num:
                        return global_sample_num
                current_event_num = line_dict['event_num']
                current_sample_num = line_dict['sample']
                line_dicts_em = []
                line_dicts_had = []
                line_dicts_tracks = []

            # if this is a cell line, append to the cells list.
            if line_dict['signal'] == '1':
                if line_dict['Had'] == '0' and line_dict['EM'] != '0':
                    line_dicts_em.append(line_dict)
                    if _SafeLog(_StripFloat(line_dict['EM'])) > normalization_data['Cells']['EM']['Max']:
                        print('Bigger EM value detected!')
                        print('The value is: ' + str(_SafeLog(_StripFloat(line_dict['EM']))))
                        print('The max is: ' + str(normalization_data['Cells']['EM']['Max']))
                elif line_dict['Had'] != '0' and line_dict['EM'] == '0':
                    line_dicts_had.append(line_dict)
                    if _SafeLog(_StripFloat(line_dict['Had'])) > normalization_data['Cells']['Had']['Max']:
                        print('Bigger had value detected!')
                        print('The value is: ' + str(_SafeLog(_StripFloat(line_dict['Had']))))
                        print('The max is: ' + str(normalization_data['Cells']['Had']['Max']))
                else:
                    print('Error: unexpected input line dict!')
                    print(line_dict)
            # if this is a track line, append to the tracks list.
            if line_dict['signal'] == '2':
                # line_dict['Had'] = ''
                line_dicts_tracks.append(line_dict)
                if _SafeLog(_StripFloat(line_dict['EM'])) > normalization_data['Tracks']['Max']:
                    print('Bigger Tracks value detected!')
                    print('The value is: ' + str(_SafeLog(_StripFloat(line_dict['EM']))))
                    print('The max is: ' + str(normalization_data['Tracks']['Max']))

    return global_sample_num


def _PackageSample(sample_dicts_em, sample_dicts_had, sample_dicts_tracks, image_size, label, normalization_data, global_sample_num, out_dir):
    """
        This function does processing on a single sample and saves it into a .pkl file.
    """
    if not sample_dicts_em or not sample_dicts_tracks or not sample_dicts_had:
        print('empty list detected!')
    # The sample is stored as a dictionary.
    outfile_data = defaultdict()
    if not sample_dicts_em or not sample_dicts_had:
        print(sample_dicts_em)
        print(sample_dicts_had)
    outfile_data['event_num'] = sample_dicts_em[0]['event_num']
    outfile_data['sample_num'] = sample_dicts_em[0]['sample']
    center = (_StripFloat(sample_dicts_em[0]['Largeeta']), _StripFloat(sample_dicts_em[0]['Largephi']))
    outfile_data['label'] = label

    # Process cell: extract numbers from strings, group all entries into a single cell collection, and turn that collection into an image.
    had_tuples = [_CellLineDictToNormalizedTuple(l, normalization_data, sensor='had') for l in sample_dicts_had]
    em_tuples = [_CellLineDictToNormalizedTuple(l, normalization_data, sensor='em') for l in sample_dicts_em]
    had_tuples = [_WrapPhis(tup, center) for tup in had_tuples]
    em_tuples = [_WrapPhis(tup, center) for tup in em_tuples]

    outfile_data['em'] = em_tuples
    outfile_data['had'] = had_tuples

    outfile_data['jet_center'] = center

    # Process tracks: extract numbers from strings, and use the bounds information from the cell to match the tracks positions with the cell position.
    tracks_tuples = [ _TrackLineDictToNormalizedTuple(l, normalization_data) for l in sample_dicts_tracks]
    # [_MatchTrackCoordsToBounds(t, bounds) for t in tracks]
    tracks_tuples = [ _WrapPhis(tup, center) for tup in tracks_tuples]
    #tracks_tuples = [_MatchTrackCoordsToCenter(tup, center) for tup in tracks_tuples]
    image, tracks_tuples = _CellToImage(em=em_tuples, had=had_tuples, tracks=tracks_tuples, use_tracks=True, image_size=image_size, jet_center=center)
    #plt.imshow(image)
    #plt.show()
    #DataVisualization.sample_visualization(image, global_sample_num)
    #exit(1)
    outfile_data['tracks'] = tracks_tuples
    outfile_data['tracks_arr'] = _TracksListToTensor(tracks_tuples)
    outfile_data['num_tracks'] = outfile_data['tracks_arr'].shape[0]
    outfile_data['cell_image'] = image
    outfile_name = out_dir + '/sample_' + str(global_sample_num).zfill(6) + '.pkl'
    with open(outfile_name, 'wb') as fo:
        pickle.dump(outfile_data, fo)


def _StripInt(s):
    return int(s.strip('"'))


def _StripFloat(s):
    return float(s.strip('"'))


def _ClampNegative(x):
    if(x < 0):
        x = 0
    return x


def _SafeLog(x):
    return log(_ClampNegative(x) + 1)


def _LogNormalize(x, min_val, max_val):
    ret_value = (_SafeLog(x) - min_val) / float(max_val - min_val)
    if ret_value > 1:
        print('Error!')
    return ret_value


def _MatchTrackCoordsToCenter(track, center):
    l = list(track)
    l[0] = l[0] - center[0]
    l[1] = l[1] - center[1]
    return tuple(l)


def _TracksListToTensor(tracks):
    eta_vec = np.array([eta for (eta, phi, Val) in tracks])
    phi_vec = np.array([phi for (eta, phi, Val) in tracks])
    Val_vec = np.array([Val for (eta, phi, Val) in tracks])
    return np.column_stack((eta_vec, phi_vec, Val_vec))


def _CellToImage(em, had, tracks, image_size, jet_center, use_tracks=True):
    eta_vec_em = np.array([eta for (eta, phi, EM) in em])
    phi_vec_em = np.array([phi for (eta, phi, EM) in em])

    eta_vec_had = np.array([eta for (eta, phi, Had) in had])
    phi_vec_had = np.array([phi for (eta, phi, Had) in had])

    eta_vec_tracks = np.array([eta for (eta, phi, track) in tracks])
    phi_vec_tracks = np.array([phi for (eta, phi, track) in tracks])

    EM_vec = np.array([EM for (eta, phi, EM) in em])
    Had_vec = np.array([Had for (eta, phi, Had) in had])
    Track_vec = np.array([track for (eta, phi, track) in tracks])
    
    eta_centered_em = (eta_vec_em - jet_center[0])
    phi_centered_em = (phi_vec_em - jet_center[1])
    eta_centered_had = (eta_vec_had - jet_center[0])
    phi_centered_had = (phi_vec_had - jet_center[1])
    eta_centered_track = (eta_vec_tracks - jet_center[0])
    phi_centered_track = (phi_vec_tracks - jet_center[1])

    eta_centered_em -= min(eta_centered_em)
    phi_centered_em -= min(phi_centered_em)

    eta_centered_em = eta_centered_em / (max(abs(eta_centered_em)) )
    phi_centered_em = phi_centered_em / (max(abs(phi_centered_em)) )
    if max(eta_centered_em) > 1 or max(phi_centered_em > 1):
        print('Violation - index will be out of bounds for EM')

    eta_centered_had -= min(eta_centered_had)
    phi_centered_had -= min(phi_centered_had)
    eta_centered_had = eta_centered_had / (max(abs(eta_centered_had)) + 1e-8)
    phi_centered_had = phi_centered_had / (max(abs(phi_centered_had)) + 1e-8)
    if max(eta_centered_had) > 1 or max(phi_centered_had > 1):
        print('Violation - index will be out of bounds for Had')
    elif min(eta_centered_had) < 0 or min(phi_centered_had) < 0:
        print('####### ERROR##########')
        print(min(eta_centered_had))
        print(min(phi_centered_had))

    eta_centered_track -= min(eta_centered_track)
    phi_centered_track -= min(phi_centered_track)
    eta_centered_track = eta_centered_track / (max(abs(eta_centered_track)))
    phi_centered_track = phi_centered_track / (max(abs(phi_centered_track)))
    if max(eta_centered_track) > 1 or max(phi_centered_track > 1):
        print('Violation - index will be out of bounds for Tracks')

    eta_indexed_em = np.floor(((eta_centered_em)) * (image_size[0] - 1)).astype(int)
    phi_indexed_em = np.floor(((phi_centered_em) ) * (image_size[1] - 1)).astype(int)

    eta_indexed_had = np.floor((eta_centered_had + 0.001) * float((image_size[0] - 1))).astype(int)
    phi_indexed_had = np.floor((phi_centered_had + 0.001) * float((image_size[1] - 1))).astype(int)
    eta_indexed_track = np.floor(((eta_centered_track) ) * (image_size[0]-1)).astype(int)
    phi_indexed_track = np.floor(((phi_centered_track) ) * (image_size[1]-1)).astype(int)

    cell_image = np.zeros((3, image_size[0], image_size[1]))
    for i in range(len(eta_vec_em)):
        cell_image[0, phi_indexed_em[i], eta_indexed_em[i]] += EM_vec[i]
        cell_image[0, phi_indexed_em[i], eta_indexed_em[i]] = min(1.0, cell_image[0, phi_indexed_em[i], eta_indexed_em[i]])
    for i in range(len(eta_vec_had)):
        cell_image[1, phi_indexed_had[i], eta_indexed_had[i]] += Had_vec[i]
        cell_image[1, phi_indexed_had[i], eta_indexed_had[i]] = min(1.0, cell_image[1, phi_indexed_had[i], eta_indexed_had[i]])
    if use_tracks:
        for i in range(len(eta_vec_tracks)):
            cell_image[2, phi_indexed_track[i], eta_indexed_track[i]] += Track_vec[i]
            cell_image[2, phi_indexed_track[i], eta_indexed_track[i]] = min(1.0, cell_image[2, phi_indexed_track[i], eta_indexed_track[i]])

    tracks_tupels = []
    for i in range(len(eta_vec_tracks)):
        tracks_tupels.append((eta_centered_track[i], phi_centered_track[i], Track_vec[i]))

    # cell_image = np.flipud(cell_image)
    return cell_image, tracks_tupels

def _WrapPhis(tup, center):
    if(center[1] + 1) > pi:      
        if tup[1] < center[1] + 1 - 2 * pi:
            l = list(tup)
            l[1] = l[1] + 2 * pi
            return tuple(l)
    if(center[1] - 1) < -pi:
        if tup[1] > center[1] - 1 + 2 * pi:
            l = list(tup)
            l[1] = l[1] - 2 * pi
            return tuple(l)
    return tup

def _GetLinecount(csv_path):
    with open(csv_path, "r") as f:
        linecount = sum(1 for _ in f)
    return linecount - 1 # exclude header line


def _CellLineDictToNormalizedTuple(line_dict, norm_data, sensor='em'):
    if sensor == 'em':
        EM_normalized = _LogNormalize(_StripFloat(line_dict['EM']), norm_data['Cells']['EM']['Min'], norm_data['Cells']['EM']['Max'])
        return (_StripFloat(line_dict['eta']), _StripFloat(line_dict['phi']), EM_normalized)
    elif sensor == 'had':
        Had_normalized = _LogNormalize(_StripFloat(line_dict['Had']), norm_data['Cells']['Had']['Min'], norm_data['Cells']['Had']['Max'])
        return (_StripFloat(line_dict['eta']), _StripFloat(line_dict['phi']), Had_normalized)
    else:
        print('Unexpected sonsor, should be em or had')


def _TrackLineDictToNormalizedTuple(line_dict, norm_data):
    Val_normalized = _LogNormalize(_StripFloat(line_dict['EM']), norm_data['Tracks']['Min'], norm_data['Tracks']['Max']) 
    return (_StripFloat(line_dict['eta']), _StripFloat(line_dict['phi']), Val_normalized)


def _ExtractNormalizationData_Running(input_signal_path, input_signal_linecount, input_bg_path, input_bg_linecount):    
    Cell_EM_data = { 'Min': float(1e10), 'Max': float(-1e10), 'Mean': 0.0, 'Std': 0.0 }
    Cell_Had_data = { 'Min': float(1e10), 'Max': float(-1e10), 'Mean': 0.0, 'Std': 0.0 }
    Tracks_value_data = { 'Min': float(1e10), 'Max': float(-1e10), 'Mean': 0.0, 'Std': 0.0 }
    out_data = { 'Cells' : { 'EM': Cell_EM_data, 'Had': Cell_Had_data}, 'Tracks': Tracks_value_data }

    em_max = float(-1e5)
    em_min = float(1e5)
    Cell_EM_sum = 0.0
    Cell_EM_SQminusmean_sum = 0.0
    Cell_count = 0.0

    had_max = float(-1e5)
    had_min = float(1e5)
    Cell_Had_sum = 0.0
    Cell_Had_SQminusmean_sum = 0.0

    tracks_max = float(-1e5)
    tracks_min = float(1e5)
    Tracks_Val_sum = 0.0
    Tracks_Val_SQminusmean_sum = 0.0
    Tracks_count = 0.0

    # First pass - min, max, mean - signal file
    print("\nFirst pass - min, max, mean - signal file...")
    with open(input_signal_path, "r") as f:
        header_line = next(f)
        categories = ['event_num', 'sample', 'signal', 'eta', 'phi', 'EM', 'Had', 'Largeeta', 'Largephi']
        # convert each csv line into a dict, and append all into a list of dicts.
        # print("\nProcessing lines...")

        for line in tqdm(f):
            # turn the line into a dictionary with key-value pairs.
            line_split = line.strip('\n').split(',')
            line_dict = defaultdict()
            i = 0
            for entry in line_split:
                line_dict[categories[i]] = entry
                i += 1

            # if this is a cell line
            if line_dict['signal'] == '1':
                Cell_count += 1
                EM_line = _SafeLog(_StripFloat(line_dict['EM']))                
                Cell_EM_sum += EM_line                                
                if EM_line > em_max:
                    em_max = EM_line
                if EM_line < em_min:
                    em_min = EM_line

                Had_line = _SafeLog(_StripFloat(line_dict['Had']))
                Cell_Had_sum += Had_line
                if Had_line > had_max:
                    had_max = Had_line
                if Had_line < had_min:
                    had_min = Had_line

            # if this is a track line
            if line_dict['signal'] == '2':
                Tracks_count += 1
                Tracks_line = EM_line = _SafeLog(_StripFloat(line_dict['EM']))               
                Tracks_Val_sum += Tracks_line                                
                if Tracks_line > tracks_max:
                    tracks_max = Tracks_line
                if Tracks_line < tracks_min:
                    tracks_min = Tracks_line

    print('EM max is: ' + str(em_max))
    print('EM min is: ' + str(em_min))
    print('HAD max is: ' + str(had_max))
    print('HAD min is: ' + str(had_min))
    print('Tracks max is: ' + str(tracks_max))
    print('Tracks min is: ' + str(tracks_min))

    # First pass - min, max, mean - background file
    print("\nFirst pass - min, max, mean - background file...")
    with open(input_bg_path, "r") as f:
        header_line = next(f)
        categories = ['event_num', 'sample', 'signal', 'eta', 'phi', 'EM', 'Had', 'Largeeta', 'Largephi']

        # convert each csv line into a dict, and append all into a list of dicts.
        # print("\nProcessing lines...")

        for line in tqdm(f):
            # turn the line into a dictionary with key-value pairs.
            line_split = line.strip('\n').split(',')
            line_dict = defaultdict()
            i = 0
            for entry in line_split:
                line_dict[categories[i]] = entry
                i += 1

            # if this is a cell line
            if line_dict['signal'] == '1':
                Cell_count += 1             
                EM_line = _SafeLog(_StripFloat(line_dict['EM']))
                Cell_EM_sum += EM_line                                
                if EM_line > em_max:
                    em_max = EM_line
                if EM_line < em_min:
                    em_min = EM_line

                Had_line = _SafeLog(_StripFloat(line_dict['Had']))
                Cell_Had_sum += Had_line
                if Had_line > had_max:
                    had_max = Had_line
                if Had_line < had_min:
                    had_min = Had_line

            # if this is a track line
            if line_dict['signal'] == '2':
                Tracks_count += 1
                Tracks_line = _SafeLog(_StripFloat(line_dict['EM']))                
                Tracks_Val_sum += Tracks_line                                
                if Tracks_line > tracks_max:
                    tracks_max = Tracks_line
                if Tracks_line < tracks_min:
                    tracks_min = Tracks_line

    print('EM max is: ' + str(em_max))
    print('EM min is: ' + str(em_min))
    print('HAD max is: ' + str(had_max))
    print('HAD min is: ' + str(had_min))
    print('Tracks max is: ' + str(tracks_max))
    print('Tracks min is: ' + str(tracks_min))

    out_data['Cells']['EM']['Min'] = em_min
    out_data['Cells']['EM']['Max'] = em_max
    out_data['Cells']['Had']['Min'] = had_min
    out_data['Cells']['Had']['Max'] = had_max
    out_data['Tracks']['Min'] = tracks_min
    out_data['Tracks']['Max'] = tracks_max
    out_data['Cells']['EM']['Mean'] = Cell_EM_sum / Cell_count
    out_data['Cells']['Had']['Mean'] = Cell_Had_sum / Cell_count
    out_data['Tracks']['Mean'] = Tracks_Val_sum / Tracks_count

    return out_data