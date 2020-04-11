# Fall Detection on the Edge and in the Cloud

Tennison Yu, Stephanie Mather, Apik Zorian, and Kevin Hanna

This project is developed as part of W251 of the MIDS program at UC Berkeley. 

## Introduction: 

According to the CDC, one out of five falls is a serious injury such as broken bones, hip fractures or head injuries. In fact, falls cause 95% of hip fractures and are the leading cause of brain-related trauma. Each year, 3 million older people (1 in 4) are treated in emergency departments for fall injuries and over 800,000 patients a year are hospitalized. In 2015, the total medical costs for falls totaled more than $50 billion in the US. Medicare and Medicaid shouldered 75% of these costs.<sup>1</sup> It has also been found that inaction after a fall, especially for older people can lead to 
pneumonia, pressure sores, dehydration, hypothermia, and even death.<sup>2</sup> Therefore prevention and early detection of falls can help reduce burden in many regards including social, personal, and economic systems and entities. 

## Project:

To help address this, we've developed fall detection software that can be implemented on devices as small as the Jetson TX2. It is based on TSM model work by Ji Lin, Ji and Chuang Gan and Song Han at MIT which can be found here: https://github.com/mit-han-lab/temporal-shift-module.

In addition, we've created a separate repository to house our training scripts. A link to that can be found below along with links to our final report and presentation:

Training Scripts: 

https://github.com/kevinhanna/temporal-shift-module

Presentation:

https://docs.google.com/presentation/d/19RWhdRCnKfmGNrZ4o836lJlcn2LB9AqKMgai3jB8BQM/edit#slide=id.g35f391192_00

Report: 

https://drive.google.com/a/berkeley.edu/file/d/1YYesC2yK389oRb568UFE7tc5Lwz5MrvS/view?usp=sharing



## Project Organization


    ├── README.md          <- The top-level README for developers using this project.
    │    │
    ├── models             <- This folder is currently empty
    │
    ├── notebooks          <- This folder is currently emptyu
    │
    ├── archives           <- sunsetted prior work and docker images 
    │
    ├── docker_images      <- Docker images for project
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- this folder is currently empty
    └── .gitignore         <- files and folders to ignore on push


## References
<sup>1</sup> https://www.cdc.gov/homeandrecreationalsafety/falls/adultfalls.html
<sup>2</sup> https://www.chicagotribune.com/lifestyles/health/sns-health-older-people-fall-research-story.html


