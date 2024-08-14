# EGT309

## Follow these steps in order

**Initial cloning:**
1) on WSL terminal, type `git clone https://github.com/chelyn2008/EGT309`

**To pull github repo without repeating the cloning:**
1) Navigate to ur local file path (e.g. cd EGT309) - current name of repo
2) type `git pull origin main` (will show changes made)

**Mounting local pc root directory (WSL) to minikube vm**
1) use `minikube start` to start the vm
2) run `minikube mount /home/chelyn_2008/EGT309/data:/mnt/data` (format: minikube mount [local directory]:[vm directory]) 
    - local dir format `/home/[githubuser]/EGT309/data` - if you cloned the repository correctly
    - change the local directory to your local WSL data directory (can be found by typing `explorer.exe .` into WSL terminal while being in cloned repo file)
3) if you mounted your local pc root dir, open a new WSL terminal after

**Pulling docker images**
- type `cd EGT309` to open cloned github repository
1) Data Processing image: `git pull rhiann/dataprocessing:latest`
2) Model Training image: `git pull chelyn/modeltraining:latest`
3) Model Inferencd image: `git pull anatasia/modelinference:latest`


**Deploying the PV (Persistent Volume), PVC (Persistent Volume Claims) & SC(Storage Class)**
- PV and PVC for data transfer between the pods
1) type `cd src` (following from the cd EGT309 above)
2) type `kubectl apply -f storageclass.yaml` to run the storage class
3) type `kubectl apply -f persistent_volume.yml` to run the pv
4) type `kubectl apply -f persistent_volume_claim.yml` to run the pvc
5) you can use `kubectl get storageclass` to check that the storage class has been created, `kubectl get pv` to check the PV(the status should be bound), as well as `kubectl get pvc` to check the PVC(status should be bound)

**Run the kubernetes cluster**
- Prerequisites: ensure that docker desktop is open and github has the most recent pushes
1) Data Processing
    - type `cd dp`
    - type `kubectl apply -f DP_deployment.yaml`
    - you can use `kubectl get pods` to observe the pods and their statuses
2) Model Training
    - to get out of the previous dir, type `cd ..`
    - type `cd mt`
    - type `kubectl apply -f MT_deployment.yaml`
3) Model Inference
    - Follow the first step of the previous number
    - type `cd mi`
    - type `kubectl apply -f MI_deployment.yaml`
  
**Looking at data in the PV (pods)**
- `kubectl exec -it <pod_name> -- ls /mnt/data`

**Checking the output/error for each application** 
- `kubectl logs -f <pod_name>`

**Checking the events, volumes and containers in a pod**
- `kubectl describe pod <pod_name>`


## Things to note (Developers)
Any changes to the `.py` files
- Must rebuild image
    - Dockerfiles are the ones with the run command for the py file
    - Rebuilding image in production env also ensures consistency and reliability

**can be deleted after seen** <br>
@rhian - pls rebuild your image and lmk again so i can try the deployments again

@mathi & ana - pls help me check through the py files and all to ensure that it matches (e.g. X_train_scaled should be saved to X_train_scaled.csv I GOT SO MAD I CLDNT DEBUG THIS AND IT WAS BCOS OF ONE STUPID MISTAKE) For u guys model inf as well, can u guys also go and help me find out what the node affinity does in PV because i tried it and it literally worked so idk some magic

note from chelyn - im sorry i couldn't give u guys more stuff! please lmk what i need to do for the slides and ill start it as soon as i wake up TvT

    
