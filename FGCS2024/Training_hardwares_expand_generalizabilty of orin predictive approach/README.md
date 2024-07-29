
# Set Up on Raspberry pi5

### Flashing the SD card 

You nedd 64/128 GB SD card ,minimum 32 GB ,Card reader 

https://youtu.be/5CBYGz_mO9U?si=c96HHxNHG34_RDjk


https://www.tomshardware.com/how-to/set-up-raspberry-pi

Refer the Blog and YouTube video

And Choose 64-Bit Rsapberry pi OS 

Set hostname: Yes
Enable SSH: yes
Use password authentication / public key: yes
Set username and password: yes
Configure wireless LAN: no
Wireless LAN country: no
Set locale settings: Yes ,imp for keyboard choose - US

and then Write 

#### after completing flash 

insert the SD card into Raspberry pi5 ,connect to screen via hdmi and put power source at least 27 watt and connect it t internet LAN port 

#### To connect it to internet via LAN

open terminal in Rpi5 OS 

```bash
sudo nmcli con add type ethernet con-name Ethernet0 ifname eth0 ip4 10.24.24.69/24 gw4 10.255.255.0
```


```bash
nmcli con mod Ethernet0 ipv4.dns "10.16.25.13 10.16.25.15"
```

```bash
nmcli con up Ethernet0 ifname eth0
```

```bash
nmcli device status
```

To verify 

Ensure NetworkManager is set to manage the eth0 interface:

```bash
nmcli device set eth0 managed yes
```

Bring up the specific network connection:
```bash
sudo nmcli c up "Conn Name"
```

If you need to restart NetworkManager to apply changes:
```bash
sudo systemctl restart NetworkManager
```


For manully change the network config, you can change the below file in nano editor:
 
```bash
sudo nano /etc/network/interfaces
```

Restart NetworkManager to apply changes made in the configuration file:
```bash
sudo service network-manager restart
```

If not worked try this

https://itslinuxfoss.com/set-up-static-ip-address-debian-12-linux/

To check connection
```bash
nmcli con show
```
