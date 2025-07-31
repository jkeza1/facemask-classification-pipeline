from locust import HttpUser, task, between
import random, os

class WebsiteUser(HttpUser):
    wait_time = between(1, 2)  # Simulate real user delay

    @task
    def predict_image(self):
        # Folders with categories
        base_path = "Dataset"
        categories = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        category = random.choice(categories)

        # Get image path
        image_folder = os.path.join(base_path, category)
        image_files = os.listdir(image_folder)
        if not image_files:
            print(f"No images in {image_folder}")
            return

        image_file = random.choice(image_files)
        image_path = os.path.join(image_folder, image_file)

        with open(image_path, 'rb') as img:
            files = {'image': (image_file, img, 'image/jpeg')}
            response = self.client.post("/predict", files=files)
            print(f"Status code: {response.status_code}, Response: {response.text}")
