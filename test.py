from vimeo_downloader import Vimeo
v = Vimeo('https://vimeo.com/503166067')
meta = v.metadata

print(meta.title)
# print(meta.likes)
# print(meta.views)
# print(meta._fields)

s = v.streams
low_stream = s[0]  # Select the best stream
print(low_stream.filesize)
print(low_stream.direct_url)
low_stream.download(download_directory='check',
                        filename='vimeo.mp4')
# List of all meta data fields

# from pytube import YouTube 
# from moviepy.editor import *
# from glob import glob

# def MP4ToMP3(mp4, mp3):
#     FILETOCONVERT = AudioFileClip(mp4)
#     FILETOCONVERT.write_audiofile(mp3)
#     FILETOCONVERT.close()


# yt = YouTube('https://www.youtube.com/watch?v=agizP0kcPjQ')
# yt.streams \
# .filter(progressive=True, file_extension='mp4') \
# .order_by('resolution') \
# .desc() \
# .first() \
# .download("test_download")

# VIDEO_FILE_PATH = glob("test_download/*.mp4")[0]
# AUDIO_FILE_PATH = "test_download/test.mp3"
# print(VIDEO_FILE_PATH)
# print(AUDIO_FILE_PATH)
# MP4ToMP3(VIDEO_FILE_PATH, AUDIO_FILE_PATH)
# # MoviePy - Writing audio in /Full/File/Path/ToSong.mp3
# # MoviePy - Done.                     