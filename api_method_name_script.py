import os
API_Folder = 'API Method txt library/'

api_names = []
for names in os.listdir(API_Folder):
	api_names.append(names[:-4])

api_methods = []
for filename in os.listdir(API_Folder):
	with open(API_Folder + filename) as f:
		methods = list(f)
		methods = [m.strip('\n') for m in methods]

	api_methods += methods

print("API NAMES")
print(api_names)
print('\n'*4)

print("METHODS")
print("NUM API METHODS:",len(api_methods))
print(api_methods[:20])