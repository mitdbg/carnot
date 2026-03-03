import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews")

print("Path to dataset files:", path)
