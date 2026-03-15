# CO-BRA-Coral-Operation--Bleaching-Risk-Assessment
Authors: Eddy Ma, Poorva Viswanaath, Andrew Choi, Conlan Lewis

Main dataset (Drive link): https://drive.google.com/file/d/18YSLKAfrOF1vdbGaKWpWBj1xrGImzeco/view?usp=sharing

##Inspiration
Coral reefs support over 25% of all ocean species. 50% of all coral reefs have been lost since 1950.

The damage to coral reefs in our oceans globally indicates a detrimental change to Earth. This is the direct result of human change in our climate over the past 70 years. Unfortunately, as climate change proceeds to accelerate, the loss of coral reefs will continue to harm our oceanic ecosystems.

Our team decided to tackle this issue while looking at ways to indicate to organizations when the death of reefs may occur, as indicated through the organisms developing heat stress, when ocean temperatures rise 1-2 degrees celsius for extended periods of times. Identifying these risk zones weeks in advance can allow efforts to be deployed sooner, and therefore, have a greater impact on the survival of these reefs. 

In issues like this, where we have already lost oceanic organisms, the same organisms that give many species the ability to thrive, time is not in our favor and waiting until after these organisms suffer heat stress does not provide any chance for organizations to step in. Our team had to change that.

##What it does

**Cobra** is a ML-powered system that is trained on decades of past information on Sea Surface Temperature and bleaching events in 38 coral reefs to assess the risk of coral bleaching 4-6 weeks in advance. **Cobra** displays every coral reef and its associated severity based on past risks on a map to display where help is mostly needed. Additionally, we have included a NOAA risk alert system that shows how bad these bleaching events are going to be.

By predicting these issues, **Cobra** does something no other system has yet been able to do, allow organizations such as the World Meteorological Organization to have insight 4-6 weeks before we risk the loss of coral reefs and prioritize the reefs with the highest risk

Boasting a success rate of 93%, **Cobra** provides a system that can undoubtedly save not only numerous coral reefs over decades of time–but also the organisms that rely on these reefs for shelter every day.

##How we built it
We began by downloading decades of daily ocean data from NOAA Coral Reef Watch. This included sea surface temperature, heat stress scores, temperature outliers/anomalies from 38 reef monitoring stations around the world.

We then moved onto feature engineering where we transformed the raw data (i.e. temperatures, months, etc.) into about 50 different signals per reef. This included the averages and peak temperatures across 7, 14, 30, 60, and 90 days, rate of heat stress rising and accumulation, and counts of how many continuous days that the temperatures were in dangerous thresholds. The goal in this step was to extract as much information that correlates/causes/hints at possible bleaching events rather than just looking at one variable. At this point, each reef is represented by a feature vector with all the features we just extracted. Then for training they are labeled with a 1 if bleaching or 0 if healthy.

We then trained a XGBoost model using these features on all data before 2020. Everything after 2020 was held out to use as the test set. Essentially it works as an ensemble or collection of decision trees. In each tree, the algorithm repeatedly selects a feature and a threshold that best splits the data into their two classes (1, bleaching or 0, healthy). This process continues across the 10,000 decision trees until the model learns a ‘set of rules’ that determine if a reef will or will not bleach.

The model’s main purpose is given today’s conditions of a reef, will a bleaching event occur within the next 3, 7, 21 days? 

Finally, we ran the predictions for the next 1, 3, 7, 30, and 42 days for each coral reef and hard coded the predictions into a map. The final HTML script shows a map where every reef is shown as a color coded dot from green to red with the model’s prediction (% bleaching risk) and the NOAA alert level.

##Challenges we ran into
The biggest problem we ran into within this project was getting our project off the ground. The planning stages went smoothly, but once we started to code, we immediately ran into a wall. We couldn't find a suitable IDE which supported group collaboration; hour after hour, our group started to grow impatient and with nothing done, our morale started to slip. Despite the time we wasted, through our struggle, we grew as a team. With each problem we tackled, our bond grew stronger, which allowed us to push through and deliver our project.

##What we're proud of
-A machine learning algorithm with a 93% accuracy.
-The intractable map within our website.
-Working together as a cohesive team.
-Learning new modules like sklearn, pickle, and many more.
-Working our softest.

##What we learned
For **Cobra** to work we needed to use many modules none of us had used prior to this hackathon. Modules like sklearn and pickle were a necessity and so we spent a large amount of time on the first day becoming acquainted with those modules and many more to create **Cobra**. We also learned how to code in harmony and divvy up the difficult tasks to work harmoniously. We have grown and learned much from this experience and are proud of what we have created.

##What's next for Cobra - Coral Operation–Bleaching Risk Assessment
**Cobra** is a program that could help mitigate the damage done on coral reefs in future years. Coral reefs are a vital part of the biosphere and they are slowly dying. Things like **Cobra** could show areas in high risk and people using **Cobra** could  then help those injured ecosystems. This would all be mitigation and not prevention as **Cobra** can’t stop bleaching, only notify when and where it will occur.

