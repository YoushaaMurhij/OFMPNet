#!/bin/bash
docker exec -it --user "user" ofmpnet \
    /bin/bash -c "
    export PYTHONPATH=\"${PYTHONPATH}:/home/user/ofmpnet\";
    cd /home/user/ofmpnet;
    /bin/bash"

