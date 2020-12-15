# Crack-Detection
Traditional CV method for detecting cracks on roads

Method:
1. grayscale
2. bilateral and gaussian filtering to denoise. Median filter to remove salt-and-pepper "noise" in the asphalt.
3. image log (+normalize) to emphasize dark cracks 
4. canny edge detection
5. morphological close filter to connect blobs
6. pick biggest blobs to clean up the result

Notes:
Biggest hurdle is finding a way to ignore all the background noise that is present in the
asphalt or cement. Could probably fine-tune the filter params to achieve better results.
Need better ways to handle shadows, lane paintings, etc.
