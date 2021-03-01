import sys
import os
import json
import dill
from collections import defaultdict
import numpy as np
import pandas as pd
from pandas import ExcelFile

# define bad students ids 
# for which recording is bad or segment annotation doesn't exist
bad_ids = {}
bad_ids['middle'] = [29429, 32951, 42996, 43261, 44627, 56948, 39299, 39421, 41333, 42462, 43811, 44319, 61218, 29266, 33163]
bad_ids['symphonic'] = [33026, 33476, 35301, 41602, 52950, 53083, 46038, 33368, 42341, 51598, 56778, 56925, 30430, 55642, 60935]

class DataUtils(object):
    """
    Class containing helper functions to read the music performance data from the FBA folder
    """

    def __init__(self, path_to_annotations, path_to_audio, band, instrument):
        """
        Initialized the data utils class
        Arg:
                path_to_annotations:	string, full path to the folder containing the FBA annotations
                path_to_audio:          string, full path to the folder containing the FBA audio
                band:					string, which band type
                instrument:				string, which instrument
        """
        self.path_to_annotations = path_to_annotations
        self.path_to_audio = path_to_audio
        self.band = band
        self.instrument = instrument
        self.bad_ids = bad_ids[band]

    def get_excel_file_path(self, year):
        """
        Returns the excel file name containing the student performance details
        Arg:
                year:	string, which year
        """
        if self.band == 'middle':
            file_name = 'Middle School'
        elif self.band == 'concert':
            file_name = 'Concert Band Scores'
        elif self.band == 'symphonic':
            file_name = 'Symphonic Band Scores'
        else:
            raise ValueError("Invalid value for 'band'")
        xls_path = 'FBA' + year + '/'
        xls_file_path = self.path_to_annotations + xls_path + file_name + '.xlsx'
        return xls_file_path

    def get_audio_folder_path(self, year):
        """
        Returns the full path to the root folder containing all the FBA audio files
        Arg:
                year:   string, which year
        """
        if self.band == 'middle':
            folder_name_band = 'middleschool'
        elif self.band == 'concert':
            folder_name_band = 'concertband'
        elif self.band == 'symphonic':
            folder_name_band = 'symphonicband'
        else:
            raise ValueError("Invalid value for 'band'")
        
        if year == '2013':
            folder_name_year = '2013-2014'
        elif year == '2014':
            folder_name_year = '2014-2015'
        elif year == '2015':
            folder_name_year = '2015-2016'
        else:
            raise ValueError("Invalid value for 'year'")
        
        audio_folder = self.path_to_audio + folder_name_year + '/' + folder_name_band
        if year == '2013':
            audio_folder += 'scores/'
        else:
            audio_folder += '/'
        return audio_folder

    def get_anno_folder_path(self, year):
        """
        Returns the full path to the root folder containing all the FBA segment files and
		assessments
        Arg:
                year:	string, which year
        """
        if self.band == 'middle':
            folder_name = 'middleschool'
        elif self.band == 'concert':
            folder_name = 'concertband'
        elif self.band == 'symphonic':
            folder_name = 'symphonicband'
        else:
            raise ValueError("Invalid value for 'band'")
        annotations_folder = self.path_to_annotations + 'FBA' + year + '/' + folder_name
        if year == '2013':
            annotations_folder += 'scores/'
        else:
            annotations_folder += '/'
        return annotations_folder
    
    def scan_student_ids(self, year):
        """
        Returns the student ids for the provide inputs as a list
        Args:
                year:	string, which year
        """
        # get the excel file path
        file_path = self.get_excel_file_path(year)

        # read the excel file
        xldata = pd.read_excel(file_path)
        instrument_data = xldata[xldata.columns[0]]
        # find the index where the student ids start for the input instrument
        start_idx = 0
        while instrument_data[start_idx] != self.instrument:
            start_idx += 1
        # iterate and the store the student ids
        student_ids = []
        while isinstance(instrument_data[start_idx + 1], int):
            student_ids.append(instrument_data[start_idx + 1])
            start_idx += 1
        # remove bad student ids
        for i in range(len(self.bad_ids)):
            if self.bad_ids[i] in student_ids: 
                student_ids.remove(self.bad_ids[i])
        return student_ids

    def get_segment_info(self, year, segment, student_ids=[]):
        """
        Returns the segment info for the provide inputs as a list of tuples (start_time, end_time)
        Args:
                year:			string, which year
                segment:		string, which segment
                student_ids:	list, containing the student ids., if empty we compute it within this
								 function
        """
        annotations_folder = self.get_anno_folder_path(year)
        segment_data = []
        if student_ids == []:
            student_ids = self.scan_student_ids(year)
        for student_id in student_ids:
            segment_file_path = annotations_folder + \
                str(student_id) + '/' + str(student_id) + '_segment.txt'
            file_info = [line.rstrip('\n')
                         for line in open(segment_file_path, 'r')]
            segment_info = file_info[segment]
            if sys.version_info[0] < 3:
                to_floats = map(float, segment_info.split('\t'))
            else:
                to_floats = list(map(float, segment_info.split('\t')))
            # convert to tuple and append
            segment_data.append((to_floats[0], to_floats[0] + to_floats[1]))
        return segment_data

    def get_pitch_contours_segment(self, year, segment_info, student_ids=[]):
        """
        Returns the pitch contours for the provide inputs as a list of np arrays
                assumes pyin pitch contours have already been computed and stored as text files
        Args:
                year:			string, which year
                segment:		string, which segment
                student_ids:	list, containing the student ids., if empty we compute it within this
								 function
        """
        data_folder = self.get_anno_folder_path(year)
        if student_ids == []:
            student_ids = self.scan_student_ids(year)
        pitch_contour_data = []
        idx = 0
        for student_id in student_ids:
            pyin_file_path = data_folder + \
                str(student_id) + '/' + str(student_id) + \
                '_pyin_pitchtrack.txt'
            lines = [line.rstrip('\n') for line in open(pyin_file_path, 'r')]
            pitch_contour = []
            start_time, end_time = segment_info[idx]
            idx = idx + 1
            for x in lines:
                if sys.version_info[0] < 3:
                    to_floats = map(float, x.split(','))
                else:
                    to_floats = list(map(float, x.split(',')))
                timestamp = to_floats[0]

                if timestamp < start_time:
                    continue
                else:
                    if timestamp > end_time:
                        break
                    else:
                        pitch = to_floats[1]
                        pitch_contour.append(to_floats[1])
            pitch_contour = np.asarray(pitch_contour)
            pitch_contour_data.append(pitch_contour)

        return pitch_contour_data

    def get_audio_file_path(self, year, student_ids=[]):
        """
        Returns the audio paths for the provide inputs as a list of strings
        Args:
                year:           string, which year
                student_ids:    list, containing the student ids., if empty we compute it within this
                                 function
        """
        data_folder = self.get_audio_folder_path(year)
        if student_ids == []:
            student_ids = self.scan_student_ids(year)
        audio_file_paths = []
        for student_id in student_ids:
            audio_file_path = data_folder + \
                str(student_id) + '/' + str(student_id) + '.mp3'
            audio_file_paths.append(audio_file_path)

        return audio_file_paths

    def get_perf_rating_segment(self, year, segment, student_ids=[]):
        """
        Returns the performane ratings given by human judges for the input segment as a list of
		tuples
        Args:
                year:			string, which year
                segment:		string, which segment
                student_ids:	list, containing the student ids., if empty we compute it within
								 this function
        """
        annotations_folder = self.get_anno_folder_path(year)
        perf_ratings = []
        if student_ids == []:
            student_ids = self.scan_student_ids(year)

        for student_id in student_ids:
            ratings_file_path = annotations_folder + \
                str(student_id) + '/' + str(student_id) + '_assessments.txt'
            file_info = [line.rstrip('\n')
                         for line in open(ratings_file_path, 'r')]
            segment_ratings = file_info[segment]
            if sys.version_info[0] < 3:
                to_floats = map(float, segment_ratings.split('\t'))
            else:
                to_floats = list(map(float, segment_ratings.split('\t')))
            # convert to tuple and append
            perf_ratings.append(
                (to_floats[2], to_floats[3], to_floats[4], to_floats[5]))

        return perf_ratings
    
    def create_data(self, year, segment, audio=False):
        """
        Creates the data representation for a particular year
        Args:
                year:           string, which year
                segment:		string, which segment
        """
        if audio:
            import librosa
        perf_assessment_data = []
        student_ids = self.scan_student_ids(year)
        segment_info = self.get_segment_info(year, segment, student_ids)
        pitch_contour_data = self.get_pitch_contours_segment(year, segment_info, student_ids)
        audio_file_paths = self.get_audio_file_path(year, student_ids)
        ground_truth = self.get_perf_rating_segment(year, segment, student_ids)
        idx = 0
        for student_idx in range(len(student_ids)):
            assessment_data = {}
            assessment_data['year'] = year
            assessment_data['band'] = self.band
            assessment_data['instrumemt'] = self.instrument
            assessment_data['student_id'] = student_ids[student_idx]
            assessment_data['segment'] = segment
            if audio == False:
                assessment_data['pitch_contour'] = pitch_contour_data[student_idx]
            else:
                y,sr = librosa.load(audio_file_paths[student_idx], offset=segment_info[idx][0], duration=segment_info[idx][1] - segment_info[idx][0])
                assessment_data['audio'] = (y,sr)
            assessment_data['ratings'] = ground_truth[student_idx]
            assessment_data['class_ratings'] = [round(x * 10) for x in ground_truth[student_idx]]
            perf_assessment_data.append(assessment_data)
            idx += 1
        return perf_assessment_data