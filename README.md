
![80x_speed](https://github.com/PeterHolderrieth/RobotLibrarian/assets/57487578/1f93efe0-6e30-486a-926f-f572d9096b06)

# Robot Librarian

Libraries around the globe store the world’s most critical human knowledge, but require laborious work to keep the books organized and sorted. The ability to use robots to do the manual labor of sorting these books would allow librarians to focus on research and core archival work. In this project, we aim to build a mobile manipulator that supports a librarian in their work and sorts books from a table into the correct shelf. Working in PyDrake, we created a virtual library environment containing shelves and a table with books on it. Using geometric perception combining ICP and graph-based clustering, a perception system recognizes the books positions on the table. Antipodal grasps are sampled, scored and ranked. Using RRT planning and inverse kinematics, actuator trajectories are found. In addition, we included further optimizations defining waypoints (e.g. right in front of the shelf) to allow planning to be more efficient. We want to  **thank professor Russ Tedrake for his wonderful lecture and teaching** as part of MIT's Robotic Manipulation course!

This project was part of the robotic manipulation

[![Watch the video(<img width="579" alt="Screenshot 2023-12-11 at 10 25 07 PM" src="https://github.com/PeterHolderrieth/RobotLibrarian/assets/57487578/e85bf809-6fca-44c1-95be-f503df384faa">)](https://www.youtube.com/watch?v=U6YdzsNt3tA)




