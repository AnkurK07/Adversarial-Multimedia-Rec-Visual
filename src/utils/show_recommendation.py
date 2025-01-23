import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

class RecommendedItemsDisplay:
    def __init__(self, rec_item, df1, user_id):
        """
        Initializes the RecommendedItemsDisplay class.

        Args:
            rec_item: A list of item IDs.
            df1: DataFrame containing item IDs and image paths.
            user_id: The user ID to personalize the plot title.
        """
        self.rec_item = rec_item
        self.df1 = df1
        self.user_id = user_id

    def show_recommended_images(self):
        """Displays images for recommended items in a grid layout with clickable links using Plotly."""
        images = []
        titles = []
        links = []

        for item_id in self.rec_item:
            try:
                # Fetch image path from DataFrame
                image_path = self.df1.loc[self.df1['item_id'] == item_id, 'image'].iloc[0]

                # Fetch the image from the URL
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))

                # Append image and title with link
                images.append(px.imshow(image).data[0])
                product_link = f"https://www.amazon.com/dp/{item_id}"
                titles.append(f"<a href='{product_link}' target='_blank'> Click here for Amazon </a>")
            except (requests.exceptions.RequestException, ValueError, IOError, IndexError) as e:
                print(f"Error displaying image for item {item_id}: {e}")

        # Create subplots
        num_items = len(self.rec_item)
        num_columns = 3
        num_rows = -(-num_items // num_columns)  # Ceiling division

        fig = make_subplots(rows=num_rows, cols=num_columns, subplot_titles=titles)

        for i, image_data in enumerate(images):
            row = i // num_columns + 1
            col = i % num_columns + 1
            fig.add_trace(image_data, row=row, col=col)

        # Update layout to increase subplot size and hide axes
        fig.update_layout(
            height=num_rows * 400,  # Increase height of each subplot
            width=1200,            # Increase total width of the plot
            title_text=f"Recommended Items for user {self.user_id}",
            showlegend=False
        )
        fig.update_xaxes(visible=False)  # Hide x-axis
        fig.update_yaxes(visible=False)  # Hide y-axis
        return fig
