# Prediction of work status for people polled in UK 2011

### Conundrum 7

Data about over five hundred thousand people polled in the UK 2011 census. It includes each persons sex, age, marital status and other points. In this challenge, we use all of this information to predict if every person is currently in work - either Employed or Self-Employed. 

An exhaustive EDA analysis is elaborated in [Exploratory Data Analysis](`exploratory_data_analysis.ipynb`)., where we develop a story-telling around the importance of the Census 2011 in the UK and their results by regions, social status, industry, health conditions and religion, just to mention some of the main features that help us to figure out the huge picture about the UK population.

The supervised machine learning problem is solved using Categorical Naive Bayes and Feature Selection techniques to improve accuracy and avoid overfitting the model. More details can be found in [Predictive Modeling of Work Status](predictive_modeling_work_status.ipynb).


### Variable List
An essential list of all the variables, classifications and codes in the microdata teaching file from the 2011 Census in England and Wales.

#### Person ID
Unique reference ID	

#### Region
10 types:
- E12000001   North East 
- E12000002   North West 
- E12000003   Yorkshire and the Humber 
- E12000004   East Midlands 
- E12000005   West Midlands 
- E12000006   East of England 
- E12000007   London 
- E12000008   South East 
- E12000009   South West
- W92000004   Wale

#### Residence
2 types:
- C  Resident in a communal establishment
- H  Not resident in a communal establishment

#### Family composition	
6 types (also -9):
- 1. Not in a family 
- 2. Married/same-sex civil partnership couple family
- 3. Cohabiting couple family 
- 4. Lone parent family (male head) 
- 5. Lone parent family (female head) 
- 6. Other related family
- -9. No code required (Resident of a communal establishment, students or schoolchildren living away during term-time, or a short-term resident)

#### Population base
3 types:
- 1. Usual resident 
- 2. Student living away from home during term-time 
- 3. Short-term resident

#### Sex
2 types:
- 1. Male 
- 2. Female 

#### Age	
8 types:
- 1. 0 to 15 
- 2. 16 to 24 
- 3. 25 to 34 
- 4. 35 to 44 
- 5. 45 to 54 
- 6. 55 to 64 
- 7. 65 to 74 
- 8. 75 and over

#### Marital status	
5 types:
- 1. Single (never married or never registered a same-sex civil partnership) 
- 2. Married or in a registered same-sex civil partnership 
- 3. Separated but still legally married or separated but still legally in a same-sex civil partnership 
- 4. Divorced or formerly in a same-sex civil partnership which is now legally dissolved 
- 5. Widowed or surviving partner from a same-sex civil partnership

#### Student (Schoolchild or full-time student)	  	
2 types:
- 1. Yes 
- 2. No

#### Country of birth	
2 types (also -9):
- 1. UK 
- 2. Non UK
- -9. No Code required (Students or schoolchildren living away during term-time)

#### Health (General health)
5 types (also -9):	
- 1. Very good health 
- 2. Good health 
- 3. Fair health 
- 4. Bad health 
- 5. Very bad health
- -9. No code required (Students or schoolchildren living away during term-time)

#### Ethnic group	
5 types (also -9):	
- 1. White 
- 2. Mixed 
- 3. Asian and Asian British 
- 4. Black or Black British 
- 5. Chinese or Other ethnic group
- -9. No code required (Not resident in England or Wales, students or schoolchildren living away during term-time)


#### Religion	
9 types (also -9):
- 1. No religion 
- 2. Christian 
- 3. Buddhist 
- 4. Hindu 
- 5. Jewish 
- 6. Muslim 
- 7. Sikh 
- 8. Other religion 
- 9. Not stated
- -9. No code required (Not resident in England or Wales, students or schoolchildren living away during term-time)

#### Economic activity	
9 types (also -9):
- 1. Economically active: Employee 
- 2. Economically active: Self-employed 
- 3. Economically active: Unemployed 
- 4. Economically active: Full-time student 
- 5. Economically inactive: Retired 
- 6. Economically inactive: Student 
- 7. Economically inactive: Looking after home or family 
- 8. Economically inactive: Long-term sick or disabled 
- 9. Economically inactive: Other
- -9. No code required (Aged under 16 or students or schoolchildren living away during term-time)


#### Occupation	
9 types (also -9):
- 1. Managers, Directors and Senior Officials 
- 2. Professional Occupations 
- 3. Associate Professional and Technical Occupations 
- 4. Administrative and Secretarial Occupations 
- 5. Skilled Trades Occupations 
- 6. Caring, Leisure and Other Service Occupations 
- 7. Sales and Customer Service Occupations 
- 8. Process, Plant and Machine Operatives 
- 9. Elementary Occupations
 	 
- -9. No code required (People aged under 16, people who have never worked and students or schoolchildren living away during term-time)

#### Industry	
12 types (also -9):
- 1. Agriculture, forestry and fishing 
- 2. Mining and quarrying; Manufacturing; Electricity, gas, steam and air conditioning system; Water supply 
- 3. Construction 
- 4. Wholesale and retail trade; Repair of motor vehicles and motorcycles
- 5. Accommodation and food service activities 
- 6. Transport and storage; Information and communication 
- 7. Financial and insurance activities; Intermediation 
- 8. Real estate activities; Professional, scientific and technical activities; Administrative and support service activities 
- 9. Public administration and defence; compulsory social security 
- 10. Education 
- 11. Human health and social work activities 
- 12. Other community, social and personal service activities; Private households employing domestic staff; Extra-territorial organisations and bodies
- -9. No code required (People aged under 16, people who have never worked, and students or schoolchildren living away during term-time) 

#### Hours worked per week	
4 types (also -9):
- 1. Part-time: 15 or less hours worked 
- 2. Part-time: 16 to 30 hours worked 
- 3. Full-time: 31 to 48 hours worked 
- 4. Full-time: 49 or more hours worked
- -9. No code required (People aged under 16, people not working, and students or schoolchildren living away during term-time)

#### Approximated social grade	
4 types (also -9)	
- 1. AB 
- 2. C1 
- 3. C2 
- 4. DE
- -9. No code required (People aged under 16, people resident in communal establishments, and students or schoolchildren living away during term-time)

Source: [Office for National Statistics licensed under the Open Government Licence v.1.0.](https://www.ons.gov.uk/census/2011census/2011censusdata/censusmicrodata/microdatateachingfile/variablelist)