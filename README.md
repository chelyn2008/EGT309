# EGT309
#Instructions
- Configure minikube directory
1) Ensure Docker Desktop is open and running on the PC
2) Go into WSL terminal
3) Type in 'minikube start' to launch minikube
4) Then type 'minikube ssh' to access the local machine terminal
5) Type 'sudo mkdir -p /mnt/data' to create the directory
6) Then type 'sudo chmod 777 /mnt/data' to declare the right permissions
   - 777 refers to all; owner, group and others to have all; read, write and execute permissions

To pull github repo without repeating the cloning:
1) Navigate to ur local file path (e.g. cd EGT309) - current name of repo
2) type `git pull origin main`
3) it will show all changes made
