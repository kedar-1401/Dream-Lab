
## 4.Tmux session login for Dreamlab system
in terminal ,Enters

```bash
ssh dreamlab@10.24.24.224
```

And password for dreamlab 
```bash
dream119
```

Then You will go to the ` dreamlab@dreamlab-Precision-3440 ` Terminal 

## 5.To open tmux terminal 
In ` dreamlab@dreamlab-Precision-3440 ` this terminal 

For Orin3

```bash
tmux attach -t orin3
```

For Orin4
```bash
tmux attach -t orin4
``` 

And password for both is same  
```bash
j3tsonDream
```

Then You will go to the following Tmux terminals

Orin3 `dream-orin3@dreamorin3-desktop:~/BERT `

Orin4 `dream-orin4@ubuntu:~/LSTM`

## 6.To enter into orin3 and Orin4 

In the respcetive terminal mention above Enter following command

For Orin3

```bash
ssh dream-orin3@10.24.24.56
```

For Orin4
```bash
ssh dream-orin4@10.24.24.57
``` 

And password for both is same  
```bash
j3tsonDream
```

Then You will go to the following orin device terminals in tmux

Orin3 `dream-orin3@dreamorin3-desktop `

Orin4 `dream-orin4@ubuntu `

## 7.To set the Power mode config at MAXN 

For this need to change the directory to respcetive model

For Orin3

```bash
cd BERT
```

For Orin4
```bash
cd LSTM
``` 

And Enter in respcetive folder the following command to set MAXN powermode config  
```bash
sudo nvpmodel -m 0
```

If you got any warning 
```bash
sudo reboot now
```

Orin3- Reboot automatically No problem

Orin4- Sometimes automatically but Sometimes need to do maually by pressing power button for Orin4(57) in LAB ,the left most button until the light of LAN cable goes off and again pressing until light get turn on 

#### Indicator : After reboot command we will get a reload window and then need to login for dreamlab system.if it doesn’t come restart manually 

## 8.To generating tegrastats files

```bash
sudo jtop
```
You will see CPU runnning or Dead in case of your runs Stopped at some point of running .For quiting from it press 'q' 

## 9.To mount to media storage


```bash
sudo mount /dev/nvme0n1p1 /media/ssd
```

#### If you run previos CPU frequncy then need to move the tegrastats folders of previous runs to ssd for data consestancy and counting 

For orin3 

```bash
sudo mv pm* media/ssd/<respcetive folder>
```

For Orin4 

```bash
sudo mv pm* media/ssd/LSTM_runs/runs/
```
## 10.To run the scripts

For orin3 

```bash
sudo bash script_4k.sh 
```

For Orin4 

```bash
sudo -E bash rerun_script.sh
```

#### 1) 'ctrl + b' and 'd' for exit the tmux terminal .So now the scripts are running in tmux you can do switch off ,restart,disconnect from internet anything you want on your system .Runs will not get affected 2) But you are in tmux session then you cant do above things so get exit by 'ctrl + b' and 'd' 

## 11.To count tegrastats open new terminal which is

Prev terminal :    `dreamlab@dreamlab-Precision-3440`

New Terminal : `dream-orin3@dreamorin3-desktop`

change the directory to respcetive model 
` cd BERT` or ` cd LSTM`


```bash
find . -name "*tegrastats*" | wc -l
```

or 
'ctrl + r' and then type find

#### : 13 GPU freq * 6 CPU cores * 4 MEM freq*1 CPU fre = 312 count







