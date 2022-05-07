# ADBH
Repository of the "Autonomous Driving Behaviour Understanding" project (University Maastricht)

## Setup
Install (Docker)[https://www.docker.com/] on your system and execute the following command afterwards.


CD to docker file
```$ cd your_path/DOCKER_API```

Built Docker
```$ Docker build -t python-fast-api .```

Run docker
```$ docker run -p 8000:8000 python-fast-api```

## Use

The API provides two endpoints located to

+ <http://localhost:8000/>


+ <http://localhost:8000/status>

as well as the auto-generated 

localhost/docs
