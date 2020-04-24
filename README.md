https://www.youtube.com/watch?v=FxHqHahVHV0&feature=youtu.be

I had yolo.

I needed more image dataset.

I had to take pictures and label them with a labeling program.
It was so annoying.
So I made an Auto Labeling code using yolo that I had.

If you have your own yolo weights file, yolo CG file, yolo names file, change it and apply it.
Of course, we can apply various models such as the Keras model, not yolo.

python3 main.py --imagedir=images --failimagedir=failDetectIMG --xmldir=xml

imagedir is images in folder
failimagedir is folder that sended images that failed to detect drone
xmldir is folder that generated xml file

put your yolo in yolo folder 

if you write 'enter' key, then apear next picture.
if you write 'd' key, then delete xml file and move strange image file to failfolder.
so if you see not good label image, then enter 'd' key

and if your yolo can not find object that you want to detect, my code will print fail and auto move the picture to failfolder

eventually, you just label pictures that fail to be detected


저는 드론 yolo가 있었고 더 많은 이미지데이터셋을 원했습니다.

그래서 수백장의 드론 사진을 찍었는데 라벨링하는 것이 너무 귀찮아서 

내가 가지고 있는 욜로로 라벨링을 하면 편하게 할수 있지 않을까 하여 작성해보았습니다.

처음에는 사진들 넣고 쫙 돌렸으나 디텍팅이 되지않은 사진들도 있고 이상한곳에 디텍팅한 사진들도 있어서 잘 라벨링을 하는지 모니터링하면서 진행해야했습니다.

따라서 만약에 라벨링이 잘되면 엔터를 눌러 다음 사진으로 넘어가세요. 자동으로 xml파일이 생성되어 있을겁니다. 

엔터를 누른 후 나온 다음 사진이 라벨링이 안되어 있으면 xml파일을 생성하지 않고 fail되었다고 print가 되며 그 사진은 ‘실패폴더’로 이동합니다. 
또 라벨링은 되어있는데 엉뚱한 곳을 라벨링하고 있으면 ‘d’ 키를 누르세요

d 키를 누르면 만들어진 xml파일이 삭제되고 그 사진은 ‘실패폴더’로 이동합니다.

결국 저는 탐지 못한 드론 사진만 라벨링하면 됩니다.


