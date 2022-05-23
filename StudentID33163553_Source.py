from collections import Counter
import sys, os, math
import numpy as np

#Define Global Variances
Suspicious_Init = 5
Con_Init = 50

#set value of con_init
def setConInit(value=50):
	global Con_Init
	Con_Init = value

#set the value of sus_init
def setSusInit(value=5):
	global Suspicious_Init
	Suspicious_Init = value

#Recieve the Con_init
def getConInit():
	global Con_Init
	return Con_Init

#Recieve the sus_init
def getSuspiciousInit():
	global Suspicious_Init
	return Suspicious_Init

#the nearest neigthbour algorithms.
#Input: K value from ForceK()
#Output: prediction, conf.
def TheNearest(test, data, k):
	ngthb = []
	kShortestDistType = []

	#for all training vectors
	for d in data:
		#use euclid compute distance
		dist = DistCalc(test,d)
		#get the shortest k distance
		ngthb = kShortest(dist, d, ngthb, k)

	for n in ngthb:
		#Bool is 1 or 0 weather normal or abnormal
		kShortestDistType.append(n[1][0])

	#count time of appears
	count = Counter(kShortestDistType).most_common(k)
	#if only 1 = confidence answer
	if(len(count) == 1): return count[0][0], 1
	#calculate confidence in percentage
	conf = int((count[0][1] / (count[0][1] + count[1][1]))*100)
	return count[0][0], conf

#Calculate distance by vector
def DistCalc(v1,v2):
	sum = float(0.0)
	for i in range(0,23):
		sum += math.pow( float(v1[i]) - float(v2[i]) , 2 )
	return math.sqrt(sum)


#The nearest neigthbours accracy check
#for training =(v1,v2,v3,...,vn), TheNearest(v1, (v2,v3,...,vn)), then TheNearest(v2, (v1,v3,...,vn)), then TheNearest(v3, (v1,v2,...,vn))
#testing vector not in training data
def _accuracy_TheNearest(the_trainer, k, outputFile=None):
	f = None
	if(not outputFile is None):
		f = open(outputFile, 'w')
	step = math.floor(len(the_trainer) / 10.0)
	correct = 0
	totalPercent = 0.0

	for i in range(0,len(the_trainer)):
		if(i % step == 0):
			print(str(i))
		testElement = the_trainer[i]
		mod_train = the_trainer[:i] + the_trainer[i+1:]
		prediction, confidence = classify(testElement, mod_train, k)
		if(int(prediction) == int(testElement[-1])):
			correct = correct + 1
			totalPercent = totalPercent + confidence
		del(prediction)
		del(confidence)
		del(testElement)
		del(mod_train)

	line = str(i+1) + ": " + str(correct) +" out of "+ str(len(the_trainer)) + " correct, at " + str((totalPercent/(i+1)))+"% total confidence"
	print(line)
	if(not f is None):
		f.write(line+'\n')
		f.close()
	return correct,	(totalPercent/len(the_trainer))*100

#detect suspicious
def suspicious_check(VecTest, data, k):
	high, mid, low, suspicious = 5,2,1,0
	if((len(VecTest) !=24) and (len(VecTest) !=25)): suspicious = suspicious + high
	#all values in vector
	PAR = np.sum([float(x) for x in VecTest[0:23]])/24.0
	if(PAR >4.762): suspicious = suspicious + high
	for i in range(0,24):
		if(i ==25 and not int(VecTest[i])==0 and not int(VecTest[i])==1): suspicious = suspicious + high
		if(float(VecTest[i]) <=0.0 and not i==25): suspicious = suspicious + high
		if(float(VecTest[i]) > 7.0): suspicious = suspicious + low
		if(float(VecTest[i]) > 8.0): suspicious = suspicious + mid
		if(float(VecTest[i]) > 10.0): suspicious = suspicious + high
	#if too few in training data
	if(len(data) < 9999): suspicious = suspicious + mid
	if(len(data) < 5000): suspicious = suspicious + high
	#if bad values for k
	if(k <=0): suspicious = suspicious + high
	return suspicious

# Classifies a new vector test affair user access to the nearest neightbour
def classify(VecTest, data, k):
	#run Thenearest
	prediction, confidence = TheNearest(VecTest, data, k)

	#decision is abnormal but below confidence initial then ignore
	if(int(prediction) == 1 and int(confidence) <= getConInit()):
		prediction = 0
	suspicious = suspicious_check(VecTest, data, k)
	if(suspicious >= getSuspiciousInit()):
		return 1, confidence + math.floor((100-confidence) /3)
	return prediction, confidence

#Get Maximum value for energy
def MaxValue(the_trainer):
	max = 0.0
	for i in range(0,len(the_trainer)):
		for j in range(0,len(the_trainer[i])):
			if(float(the_trainer[i][j]) > float(max)): max = the_trainer[i][j]
	return max

#squreroot in data
#if k even and 2 options, option A could equal option B, no prediction
def ForceK(data):
	k = math.floor(math.sqrt(len(data)))
	#Force to odd
	if(k%2==0): k+=1
	return k

#Input:dist point ngthb k
#Output:neighbours list, scope by k
def kShortest(dist: float, point, ngthb: [[float,str]], k: int) ->[[float,str]]:
	if(len(ngthb) < min(10,k)) :
		ngthb.append([ dist , point[-1]])
		ngthb.sort()
		return ngthb
	if(dist > ngthb[-1][0]): return ngthb
	ngthb.append([ dist , point[-1]])
	ngthb.sort()
	return ngthb[:k]

#for all testing data
def The_catagorise(testing, the_trainer, k, outputFile=None):
	labeled = []
	i = 1
	for t in testing:
		prediction, confidence = classify(t, the_trainer, k)
		labeled.append(('line'+str(i), prediction, str(confidence)+'%',t))
		line = "line "+str(i)+" = "+ str(prediction) +" @ "+ str(confidence) +'%'
		print(line)
		i = i +1
	if(not outputFile is None):
		f = open(outputFile, 'w')
		for t in [l for l in labeled]:
			for p in t[-1]:
				f.write(str(float(p))+',')
			f.write(str(int(t[1]))+'\n')
		f.close()
	return labeled

#Take file use as input and output flie.
#used for traing and testing files (24hours + normal/not)
def InFile(fileName):
	#Check file format
	if(os.path.splitext(fileName)[1] != ".txt"): return
	data = []
	#open file with readmode
	with open(fileName, mode='r', encoding='utf-8') as f:
		for line in f:
			#read line and encode then add to var
			vector = InLine(line)
			#check vector (i.e. 24 or 25)
			if(len(vec)>=24):
				#add to output var
				data.append(vector)
	return data

#vector encode be called from InFile to input the line from file.
def InLine(line):
	#Check empty
	if(line == "" or line == " "): return []
	#split data and conver to list
	vec = list(line.split(","))
	#Check valid then return
	if(len(vec) == 24 or len(vec) == 25): return vec


#main, Command to use
# 0,None-> run classify all testing
# 1-> run accuracy_TheNearest
# 2 -> confidence of abnormal results
# Filename -> to file, run classify all testing  and print to stdout + specified file
def main(get_arguments):
	#checks argruments
	if(len(get_arguments)<=1):
		print("No argruments")
		exit(0)
	the_trainer = InFile(get_arguments[1])
	testing = InFile(get_arguments[2])
	if(the_trainer is None or the_trainer is [] or testing is None or testing is []):
		print("invalid input")
		quit()
	k = ForceK(the_trainer)
	var = 0
	try:
		var = get_arguments[3]
	except:
		pass

	setConInit(53)

	if(type(var) is str and os.path.splitext(var)[-1] == ".txt"):
		labed = The_catagorise(testing, the_trainer, k, var)
		abnormal = [int(l[0][4:]) for l in labed if int(l[1]) == 1]
		print(str(len(abnormal))+" Abnormal results, on lines: "+str(abnormal))
		quit()

	var = int(var)
	if(var == 0):
		labed = The_catagorise(testing, the_trainer, k)
		abnormal = [int(l[0][4:]) for l in labed if int(l[1]) == 1]
		print(str(len(abnormal))+" Abnormal results, on lines: "+str(abnormal))
	elif(var == 1):
		_accuracy_TheNearest(the_trainer, k, "TheNearestAccuracy_Base.txt")
	else:
		pass


if __name__ == '__main__':
	main(sys.argv)
	exit(0)
