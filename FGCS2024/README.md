

# Fixing Nvidia Jetson Orin Nano Developer Kit Issue

### "The fan starts spinning when I power it on, and the boot screen shows on the monitor. When I enter the boot, the fan stops spinning, and the monitor goes dark. I tried powering off & on several times, and every time the same thing happens.”

- There might be two reasons behind this issue:

  1. The SD card is corrupted or fragmented, preventing the Orin from booting.

  2. The Jetson developer kit itself has an issue.



- To check the cause, let’s start with the SD card since it’s easier to troubleshoot. The procedure for checking any issue with the Jetson developer kit is more complicated, which I will cover later.


- First, let’s check the SD card. If you have another Jetson Nano device in your cluster, you can simply test your SD card by inserting it into another device. If the same problem occurs with the other device, the issue is with the SD card.

### Solution for SD Card Corruption 

- Our goal is to free up about 5GB of space on the SD card to resolve the corruption problem. There’s no need to delete contents from the SSD as it doesn’t typically have corruption issues and deletion won’t help.
  1) Extract Important Files: First, you need to extract files and code stored on the SD card in your Jetson Orin Nano. It’s important to note that you don’t need to extract any files from the SSD, as it’s generally used to store large datasets and isn’t likely to be affected by this issue. This step is crucial to ensure you don’t lose your work.

  2) Access Files on a Linux PC: you can easily access the files by plugging the SD card into a card reader connected to your PC.

  3) Access Files on a Windows PC: you’ll need to install a tool like ext2fsd. This tool allows Windows to read EXT2/EXT3/EXT4 partitions, enabling you to access the files on your SD card.
  4) Backup Your Files: Store the files from your SD card (typically from the Home folder) in a secure location, such as OneDrive or another preferred storage device. This ensures you have a backup of your scripts, code, and important project files.
  5) Delete Large Files: After backing up your content, you can safely delete files from the SD card. If you encounter problems deleting files due to permission issues (some files might be in read-only mode), follow the next step.
  6) Handle Read-Only Files on Windows: Download and install Linux File System for Windows from Paragon. This software helps you mount the SD card on Windows, allowing you to delete or cut and paste files into your laptop without any permission issues.

 
  https://www.paragon-software.com/home/linuxfs-windows/?source=post_page-----a7f4a865d488--------------------------------#
 
  7)  Reinsert and Boot: After clearing up space on the SD card, reinsert it into your Jetson Orin Nano and power it on to see if the issue is resolved.

- If you still have the same problem as before, then the issue lies with the Jetson Developer Kit itself.

### Solution for Issue with the Jetson Developer Kit

- We will use a Serial Debug Console which is a useful tool for embedded development, remote access, and those times when the development kit has issues that you need to observe.

- Since the Orin Nano communicates over a basic serial cable, almost any computer with serial terminal software can interact with the Jetson. There are numerous software terminal emulators available, providing a variety of options for users.

- First, we will see how to connect Jetson Orin Nano with your host system. The Jetson Orin Nano headers use TTL logic. While there are various ways to interface with it, we chose to convert the signal to USB.

- I have used SparkFun USB to TTL Serial Cable— Debug / Console Cable, You can buy it from Amazon.


  #### 1)Wiring 
  #### Make Sure that the Power is not connected to the Cable !!!!

  The serial debug console is available on the J50 header on the Jetson Orin Nano. The J50 header is located on the edge of the carrier board opposite the I/O connectors. The connector is underneath the Jetson module, directly below the SD Card reader.

  Jetson Orin Nano J50 Pin 4 (TXD) → Cable RXD (Brown Wire)\
  Jetson Orin Nano J50 Pin 3 (RXD) → Cable TXD (Orange Wire)\
  Jetson Orin Nano J50 Pin 7 (GND) → Cable GND (Black Wire)

  The Jetson Orin Nano J50 pins are also silkscreened on the underside of the board. Here’s what it should look like: 

  If you have another cable which is most likely used

  Jetson Orin Nano J50 Pin 4 (TXD) → Cable RXD (White Wire)\
  Jetson Orin Nano J50 Pin 3 (RXD) → Cable TXD (Green Wire)\
  Jetson Orin Nano J50 Pin 7 (GND) → Cable GND (Black Wire)

  #### 2) Software 

  Before you can connect with a serial terminal application on the other computer, you will need to determine the port to which the Jetson Nano connects. This is dependent on the computer you are using.

  For Ubuntu: 
  1. To see the serial device connected via USB

  ```bash
  dmesg --follow
  ```
  Next, plug in the Serial Cable via USB other end of the Jumper wires. You will see a driver assign the cable a port number.

  2. Install Minicom: Open the terminal and run the following command:

  ```bash
  sudo apt-get install minicom
  ```
  3. To set up the Minicom app 

  The -s option is used to set up Minicom. Type the following command at the shell prompt:

  ```bash
  minicom -s
  ```

  4. The Minicom app keyboard shortcut keys

  Use the following keys:

      1. UP arrow-up or k
      2. DOWN arrow-down or j
      3. LEFT arrow-left or h
      4. RIGHT arrow-right or l
      5. CHOOSE (select menu) Enter
      6. CANCEL ESCape

  5. Serial Port Setup

  Enter the Serial Port Setup via the above keys and change the serial Device to ‘ /dev/ttyUSB0 ’, As we got in dmesg command output
    
      1. Press A to set the serial device name
      2. Press E to set Bps/Par/Bits
      3. Press [ESC] to exit
      4. Save setup as DFL
      5. Exit

  Then you will get a message of Configuration saved Then Exit via the last opt.

  After completion of this task, exit Minicom and restart to have the settings take effect.

  ```bash
  sudo minicom
  ```

  Now power on the Jetson Orin Nano by plugging in Charger, at which point you will see the kernel log starting to scroll on the Minicom window on the host.

  There are a wide variety of ways to interact with the Nano through the serial console, one of the more useful tips is to interrupt the startup process with a keystroke to be able to interact with the bootloader. You can also choose different kernels during this process when they are present. If you do not interrupt the process with a keystroke, the Nano will boot and you will be at a Terminal prompt. 

  If you’re as unlucky as Frane Selak, you might run into errors during this process. The main goal here is to capture logs that can help figure out what’s stopping your Jetson Orin Nano from working correctly.

  Here is an example of a serial log I captured: serial_console_log.txt.'[https://forums.developer.nvidia.com/uploads/short-url/uQNRfl76KoXG2UONUZ03RWCWCLC.txt']

  If you receive a similar output, it indicates that the issue might not be with the SD card. In such cases, we need to re-flash the Jetson Orin Nano device using the SDK Manager, as simply re-flashing the SD card may not resolve the issue.
