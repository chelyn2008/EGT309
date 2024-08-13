# EGT309

## Follow these steps in order

**Initial cloning:**
1) on WSL terminal, just run `git clone https://github.com/chelyn2008/EGT309`

**To pull github repo without repeating the cloning:**
1) Navigate to ur local file path (e.g. cd EGT309) - current name of repo
2) type `git pull origin main`
3) it will show all changes made

**Mounting local pc root directory (WSL) to minikube vm**
1) use `minikube start` to start the vm
2) run `minikube mount /home/chelyn_2008/EGT309/data:/mnt/data` (format: minikube mount [local directory]:[vm directory]) 
    - change the local directory to your local WSL data directory (can be found by typing `explorer.exe .` into WSL terminal while being in cloned repo file)
3) if you mounted your local pc root dir, open a new WSL terminal after

**Pulling docker images**
- type `cd EGT309` to open cloned github repository
1) Data Processing image: `git pull rhiann/dataprocessing:latest`
2) Model Training image: `git pull chelyn/modeltraining:latest`
3) Model Inferencd image: `git pull anatasia/modelinference:latest`

**Deploying the PV (Persistent Volume) & PVC (Persistent Volume Claims)
- PV and PVC for data transfer between the pods
1) type `cd src` (following from the cd EGT309 above)
2) type `kubectl apply -f persistent_volume.yml`
3) you can use `kubectl get pods` to observe the pods and their statuses

**Run the kubernetes cluster**
- Prerequisites: ensure that docker desktop is open and github has the most recent pushes
1) Data Processing
    - type `cd dp`
    - type `kubectl apply -f DP_deployment.yaml`
    - you can use `kubectl get pods` to observe the pods and their statuses
    