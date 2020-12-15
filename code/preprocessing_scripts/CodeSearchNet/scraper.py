import bs4 as bs
import urllib.request
import string 
import json
import time
def main():

	start = time.time()
	
	urlist=[]
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


	for iterVar in finalList:
		if iterVar["path"] not in urlDict:
			urlDict[iterVar["path"]]=iterVar["url"]

	for key,value in urlDict.items():

		returnFinalDict.append(value)
	return returnFinalDict

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