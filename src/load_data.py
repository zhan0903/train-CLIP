from minedojo.data import YouTubeDataset
import re
# from IPython.display import YouTubeVideo


youtube_dataset = YouTubeDataset(
        full=False,     # full=False for tutorial videos or 
                       # full=True for general gameplay videos
        download=True, # download=True to automatically download data or 
                       # download=False to load data from download_dir
        download_dir="./youtube_dataset"
                       # default: "~/.minedojo". You can also manually download data from
                       # https://doi.org/10.5281/zenodo.6641142 and put it in download_dir.           
        )

print(len(youtube_dataset))
youtube_item = youtube_dataset[0]
for item in youtube_dataset:
    if "milk" in item["title"].casefold():
        print(item)
        # break

# print(youtube_item.keys())
# print(youtube_item.values())