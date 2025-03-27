from gradio_client import Client, handle_file

client = Client("pablovela5620/mini-dust3r")
result = client.predict(
		image_name_list=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
		api_name="/predict"
)
print(result)