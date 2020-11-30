from bs4 import BeautifulSoup
import urllib.request

html_page = urllib.request.urlopen("https://pandas.pydata.org/docs/reference/style.html")
soup = BeautifulSoup(html_page, "html.parser")
PATH = "https://pandas.pydata.org/docs/reference/"
all_links = list(soup.findAll('a'))[20:]
for i,link in enumerate(all_links):
	full_path = PATH + link.get('href')
	try:
		new_links = link.get('href').split('/')
		val = new_links[1][:-5]
		html_page2 = urllib.request.urlopen(full_path)
		soup2 = BeautifulSoup(html_page2, "html.parser")
		all_results = soup2.findAll('dt', {'id': val})
		actual_stuff = all_results[0].text[:-1].strip()
		if actual_stuff[-8:] == '[source]':
			actual_stuff = actual_stuff[:-8]

		if actual_stuff.split(' ')[0] in ['class','classmethod','property']:
			actual_stuff = ' '.join(actual_stuff.split(' ')[1:])


		if actual_stuff[:6] != 'pandas':
			path = '.'.join(val.split('.')[:3])
			actual_stuff = path.replace('pandas','pd') + '.' + actual_stuff[:]

		else:
			actual_stuff = 'pd' + actual_stuff[6:]
		print(actual_stuff)

	except IndexError:
		break
    

print(f"{i} many items")