import time
from pathlib import Path
from typing import Iterable, Optional

import dateutil.parser
import gcapi

from .io import PathLike

JOB_SUCCEEDED = "Succeeded"


def download_algorithm_results_from_archive(
    client: gcapi.Client,
    *,
    archive_slug: str,
    algorithm_slug: str,
    result_interface_slug: str,
    destination: PathLike,
    cooldown: int = 5,
):
    """
    Download image results (like segmentation masks) from an algorithm that is
    attached to an archive - the API is queried for a list of images in the
    archive and if the specified algorthm produced a result for those images,
    the result is downlaoded into the destination folder.

    Assumes that the algorithm input is a single image!
    """

    assert cooldown > 0

    # Get archive and algorithm IDs
    archive = client.archives.detail(slug=archive_slug)
    algorithm = client.algorithms.detail(slug=algorithm_slug)

    # Get list of images in the archive and their names
    image_ids = list()
    image_names = dict()
    for image in client.images.iterate_all(params={"archive": archive["pk"]}):
        image_ids.append(image["pk"])
        image_names[image["api_url"]] = image["name"]

    # Request algorithm job details in chunks
    chunk_size = 100
    chunks = [image_ids[i : i + chunk_size] for i in range(0, len(image_ids), chunk_size)]

    downloaded = set()
    for chunk in chunks:
        jobs = client.algorithm_jobs.page(
            limit=len(chunk),
            params={"algorithm_image__algorithm": algorithm["pk"], "input_image": chunk},
        )
        for job in jobs:
            for input in job["inputs"]:
                if input["image"] in image_names:
                    image_url = input["image"]
                    break
            else:
                continue  # should not happen

            # Skip jobs that did not complete successfully
            if job["status"] != JOB_SUCCEEDED:
                continue

            # Download
            basename = (Path(destination) / image_names[image_url]).with_suffix("")
            for result in job["outputs"]:
                if result["interface"]["slug"] == result_interface_slug:
                    client.images.download(url=result["image"], filename=basename)
                    downloaded.add(image_url)
                    time.sleep(cooldown)

    return downloaded


def add_algorithm_results_to_archive_items(
    client: gcapi.Client,
    *,
    archive_slug: str,
    algorithm_slug: str,
    result_interface_slugs: Iterable[str],
    cooldown: int = 5,
):
    """
    Find image results (like segmentation masks) from an algorithm that is
    attached to an archive and move them back into that same archive, where
    the algorithm results get associated with the original image.

    Assumes that the algorithm input is a single image!
    """

    assert cooldown > 0

    # Get archive and algorithm IDs
    archive = client.archives.detail(slug=archive_slug)
    algorithm = client.algorithms.detail(slug=algorithm_slug)

    # Get all archive items
    updated = set()
    for archive_item in client.archive_items.iterate_all(params={"archive": archive["pk"]}):
        for value in archive_item["values"]:
            if value["interface"]["kind"] == "Image":
                image = client(url=value["image"])
                break
        else:
            continue  # no image in this archive item

        # Get jobs with this input image and sort by start date
        jobs = [
            job
            for job in client.algorithm_jobs.iterate_all(
                params={"algorithm_image__algorithm": algorithm["pk"], "input_image": image["pk"]}
            )
            if job["status"] == JOB_SUCCEEDED and job["started_at"] is not None
        ]
        if len(jobs) == 0:
            continue
        jobs.sort(key=lambda job: dateutil.parser.isoparse(job["started_at"]))

        # Get algorithm outputs that match the list of component interfaces
        values = dict()
        if isinstance(result_interface_slugs, str):
            interfaces = {result_interface_slugs}
        else:
            interfaces = set(result_interface_slugs)

        for output in jobs[-1]["outputs"]:
            slug = output["interface"]["slug"]
            if slug in interfaces and output["interface"]["super_kind"] == "Image":
                values[slug] = output["image"]

        # Add algorithm results to archive item
        if len(values) > 0:
            client.update_archive_item(archive_item_pk=archive_item["pk"], values=values)
            updated.add(archive_item["pk"])

        # Wait a bit before continuing with the next request
        time.sleep(cooldown)

    return updated


def download_archive_items(
    client: gcapi.Client,
    *,
    archive_slug: str,
    interface_slug: Optional[str] = None,
    destination: PathLike,
    cooldown: int = 5,
):
    """
    Download images (like segmentation masks) from an archive. This function loops
    over the archive items in an archive and assumes that there is at least one
    image and optionally multiple segmentation masks. If no interface slug is
    specified, the image is downloaded, otherwise the matching element.
    """

    assert cooldown > 0

    # Get archive ID
    archive = client.archives.detail(slug=archive_slug)

    # Get all archive items
    downloaded = set()
    for archive_item in client.archive_items.iterate_all(params={"archive": archive["pk"]}):
        time.sleep(cooldown)

        # Identify image to be able to determine the name of the item
        for value in archive_item["values"]:
            if value["interface"]["kind"] == "Image":
                image = client(url=value["image"])
                break
        else:
            continue  # no image in this archive item

        # Search for the item to download
        if interface_slug is None:
            image_url = image["api_url"]
        else:
            for value in archive_item["values"]:
                if (
                    value["interface"]["slug"] == interface_slug
                    and value["interface"]["super_kind"] == "Image"
                ):
                    image_url = value["image"]
                    break
            else:
                continue  # no value with the right interface in this archive item

        # Download
        basename = (Path(destination) / image["name"]).with_suffix("")
        client.images.download(url=image_url, filename=basename)
        downloaded.add(image_url)

    return downloaded
