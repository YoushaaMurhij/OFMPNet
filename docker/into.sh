#!/bin/bash
docker exec -it --user "user"  ofpnet \
    /bin/bash -c "
    export PYTHONPATH=\"${PYTHONPATH}:/home/user/ofpnet\";
    cd /home/user/ofpnet;
    /bin/bash"

