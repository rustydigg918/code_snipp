import requests

def download_file(url):
	local_filename = url.split('/')[-1]

	with requests.get(url) as r:
		assert r.status_code == 200, f'error, status code is {r.status_code}'
		with open(local_filename, 'wb') as f:
			f.write(r.content)
	return local_filename

invoice = 'https://bit.ly/2UJgUpO'
inv_pdf = download_file(invoice)
