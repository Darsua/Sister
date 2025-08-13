import netifaces
import os
import ipaddress
import requests

interface = "enp1s0" # Might have to change this on a different VM

def get_ip(interface_name):
    try:
        addresses = netifaces.ifaddresses(interface_name)
        ipv4_info = addresses.get(netifaces.AF_INET)
        if ipv4_info:
            return ipv4_info[0]['addr']
        else:
            return 0
    except:
        return 0

def is_valid_cidr(ip_cidr):
    try:
        ipaddress.IPv4Interface(ip_cidr)
        return True
    except ValueError:
        return False

def dhcp():
    print("\nAttempting to obtain a dynamic address...")
    os.system(f"sudo ip a flush {interface} && sudo ifdown {interface} && sudo ifup {interface}")
    return "Success!\n" if get_ip(interface) != 0 else "Faliure!\n"

def static():
    new_addr = input("\nEnter IP in CIDR notation i.e. 192.168.100.77/24 (Input 'x' to abort): ")
    if new_addr == 'x':
        return "Setting static IP aborted.\n"
    
    while not is_valid_cidr(new_addr):
        new_addr = input("\nInvalid IP format! Please use CIDR notation i.e. 192.168.100.77/24 (Input 'x' to abort): ")
        if new_addr == 'x':
            return "Setting static IP aborted.\n"

    print("Attempting to obtain a static address...")
    os.system(f"sudo ip a flush {interface} && sudo ip a add {new_addr} dev {interface}")
    return "Success!\n" if get_ip(interface) != 0 else "Faliure!\n"
    
def do_get():
    url = input("\nURL: ")
    print(f"\nAttempting to access {url}...")
    
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        
        print(f"\n---------- {url} ----------")
        print(r.text)
        print("---------- END ----------")
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    
    input("\nPress enter to exit...")
    return ""
    
def main():
    status = ""
    choice = 0
    while True:
        os.system('clear')
        print(status, end="")
        
        ip = get_ip(interface)
        print(f"Currently connected with IP: {ip}\n") if ip else \
        print("Not currently owning an IP. Please acquire one!\n")
        
        print("1. Attempt a DHCP request for a new IP")
        print("2. Attempt to set a static IP configuration")
        print("3. Access the intranet...") if ip else 0
        print("0. Exit")
        
        choice = input(f"What will you do? (0-{3 if ip else 2}) ")
        if choice == '1':
            status = dhcp()
        elif choice == '2':
            status = static()
        elif choice == '3':
            status = do_get() if ip else "Please acquire an IP address first!\n"
        elif choice == '0':
            print("\nBye-bye!\n")
            exit(0)
        else:
            status = ""

if __name__ == "__main__":
    main()