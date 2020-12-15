import bs4 as bs
import urllib.request
import string 
import json
import time
def main():

	start = time.time()
	
	urlist=[]
	#urlist.append("https://github.com/ageitgey/face_recognition/blob/c96b010c02f15e8eeb0f71308c641179ac1f19bb/examples/face_recognition_knn.py#L46-L108")
	#urlist.append("https://github.com/ageitgey/face_recognition/blob/c96b010c02f15e8eeb0f71308c641179ac1f19bb/face_recognition/api.py#L32-L39")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/sql/types.py#L758-L820")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/ml/regression.py#L211-L222")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/shuffle.py#L71-L79")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/rdd.py#L317-L327")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/sql/column.py#L57-L66")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/mllib/feature.py#L611-L624")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/mllib/tree.py#L39-L52")
	#urlist.append("https://github.com/apache/spark/blob/618d6bff71073c8c93501ab7392c3cc579730f0b/python/pyspark/conf.py#L164-L172")
	finalDict={}
	urlist=parse()

	for urlVar in range (len(urlist)):
		finalDict=scraperUrl(urlist[urlVar],finalDict)

	for key, value in dict(finalDict).items():
		if value is None or key is None:
			del finalDict[key]
	print({k: v for k, v in sorted(finalDict.items(), key=lambda item: item[1],reverse=True)})
	
	end = time.time()
	print("{:.3f}".format(end - start), "seconds")

def parse():
	finalList = []
	returnFinalDict=[]
	urlDict={}
	print("Started Reading JSON file which contains multiple JSON document")
	with open('python_train_7.json') as f:
	    for jsonObj in f:
	        studentDict = json.loads(jsonObj)
	        finalList.append(studentDict)

	#print("Printing each JSON Decoded Object")
	for iterVar in finalList:
		if iterVar["path"] not in urlDict:
			urlDict[iterVar["path"]]=iterVar["url"]
		#print(iterVar["code_tokens"], "\n")
	for key,value in urlDict.items():
		#print(key, ' : ', value)
		returnFinalDict.append(value)
	return returnFinalDict
	#print(len(urlDict))
def scraperUrl(url,libraryDict):
	try:
		source = urllib.request.urlopen(url).read()
		soup = bs.BeautifulSoup(source,'html.parser')
		#scrapping specifically with a table example
		table = soup.table

		#find the table rows within the table
		table_rows = table.find_all('tr')
		check=False
		#libraryDict["math"]=1
		# iterate through the rows, find the td tags, and then print out each of the table data tags:
		for tr in table_rows:
			td = tr.find_all('td')
			#row=td.get_text()
			
			for i in td:
				if i.get_text().startswith( 'import' ) :
					#print(i.get_text()[7:], len(i.get_text()[7:]))
					if i.get_text()[7:] in libraryDict:
						libraryDict[i.get_text()[7:]] +=1
					else:
						libraryDict[i.get_text()[7:]]=1
				#print("This is from library")
				if i.get_text().startswith( 'from ' ):
					print("Here", url,i.get_text())
					temp=i.get_text().split("from ",1)[1]

					if temp.split()[0] in libraryDict:
						libraryDict[temp.split()[0]] += 1
					else:
						libraryDict[temp.split()[0]] = 1

	except:
		pass

	return libraryDict
if __name__ == "__main__":
	main()