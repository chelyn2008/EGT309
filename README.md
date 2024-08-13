# EGT309

Initial cloning:
1) on WSL terminal, just run `git clone https://github.com/chelyn2008/EGT309`

To pull github repo without repeating the cloning:
1) Navigate to ur local file path (e.g. cd EGT309) - current name of repo
2) type `git pull origin main`
3) it will show all changes made

Mounting local pc root directory (WSL) to minikube vm
1) use `minikube start` to start the vm
2) run `minikube mount /home/chelyn_2008/EGT309/data:/mnt/data` (format: [host directory]:[vm directory]) 
    - change the host directory to your local WSL data directory
    - can be found by typing `explorer.exe .` into WSL terminal.
