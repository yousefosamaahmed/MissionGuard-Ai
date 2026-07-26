MissionGuard AI — Get Started Here

This version runs the website, PostgreSQL, and pgAdmin simultaneously using Docker.

1) Unzip the entire file into a regular folder.

2) Open Docker Desktop and wait until "Engine running" appears.

3) Open the resulting project folder.

4) Double-click START_DOCKER_WINDOWS.bat.

5) Wait for the website and pgAdmin to open automatically.

6) After successful launch, you will find the actual links in LOCAL_LINKS.txt.

pgAdmin details:

Email: admin@missionguard.com

Password: MissionGuardAdmin2026_Local

PostgreSQL password when opening the server within pgAdmin:
MissionGuardDB2026_Local

After changing the website's appearance, run START_DOCKER_WINDOWS.bat again;

The new code will be built while preserving the database data.

Do not delete the entire Docker installation. Refer to RUN_WITH_PGADMIN_AR.txt to learn when to use STOP or RESET without deleting other Docker projects.

If the run doesn't work, send the DOCKER_STARTUP_LOG.txt file that the script will generate.