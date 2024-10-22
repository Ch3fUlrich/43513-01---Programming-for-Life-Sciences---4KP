# run gillespie simulatio module

print('initializing...')  # noqa

# Code destined to generating
# segmentation masks based on
# previously created binary masks.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from argparse import ArgumentParser
from src.classes.ProgressTracker import ProgressTracker
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "runs gillespie simulation"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # parameters param
    parser.add_argument('-p', '--parameters-file',
                        dest='parameters_file',
                        required=True,
                        help='defines path to input parameters file (.txt)')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines output folder (folder that will contain segmentation masks)')

    # skip enter param
    parser.add_argument('-s', '--skip-enter',
                        dest='skip_enter',
                        action='store_true',
                        required=False,
                        help='defines whether to suppress "Enter to continue" input before execution')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

#####################################################################
# progress tracking related functions


def get_params_from_file(file_path: str) -> dict:
    """
    Reads parameters file and returns
    parameters stored in file as a dictionary.
    """
    # TODO: check how Sergej did with yaml files cause
    #  doing it like text file is very dumb.
    # defining placeholder value for params dict
    params_dict = {}

    # opening file in read mode
    with open(file_path, 'r') as open_file:

        # getting file lines
        lines = open_file.readlines()

        # iterating over lines
        for line in lines:

            # clearing line from "enter"
            line = line.replace('\n', '')

            # getting current line split
            line_split = line.split('=')

            # extracting values
            key, value = line_split

            # converting value to number


            # assembling current dict
            current_dict = {key: value}

            # updating params dict
            params_dict.update(current_dict)

    # returning params dict
    return params_dict


class ModuleProgressTracker(ProgressTracker):
    """
    Defines ModuleProgressTracker class.
    """
    # defining ProgressTracker init
    def __init__(self) -> None:
        """
        Initializes a ModuleProgressTracker instance
        and defines class attributes.
        """
        # inheriting attributes and methods from ProgressTracker
        super().__init__()

        # defining current module specific attributes

        # step
        self.steps_num = 0
        self.current_step = 0

        # trajectory
        self.trajectories_num = 0
        self.current_trajectory = 0

    # overwriting class methods (using current module specific attributes)

    def get_progress_string(self) -> str:
        """
        Returns a formated progress
        string, based on current progress
        attributes.
        """
        # assembling current progress string
        progress_string = f''

        # checking if iterations total has already been obtained
        if not self.totals_updated:

            # updating progress string based on attributes
            progress_string += f'calculating totals...'
            progress_string += f' {self.wheel_symbol}'
            progress_string += f' | steps: {self.steps_num}'
            progress_string += f' | trajectories: {self.trajectories_num}'
            progress_string += f' | iterations: {self.iterations_num}'
            progress_string += f' | elapsed time: {self.elapsed_time_str}'

        # if total iterations already obtained
        else:

            # updating progress string based on attributes
            progress_string += f'generating masks...'
            progress_string += f' {self.wheel_symbol}'
            progress_string += f' | image: {self.current_iteration}/{self.iterations_num}'
            progress_string += f' | progress: {self.progress_percentage_str}'
            progress_string += f' | elapsed time: {self.elapsed_time_str}'
            progress_string += f' | ETC: {self.etc_str}'

        # returning progress string
        return progress_string

    def update_totals(self,
                      args_dict: dict
                      ) -> None:
        """
        Implements module specific method
        to update total iterations num.
        """
        # getting params path
        params_path = args_dict['parameters_file']

        # getting simulation params
        params_dict = get_params_from_file(file_path=params_path)

        print(params_dict)
        self.exit()

        # iterating over files
        for _ in range(10):

            # updating progress tracker attributes
            self.iterations_num += 1

        # updating totals string
        totals_string = f'totals...'
        totals_string += f' | steps: {self.steps_num}'
        totals_string += f' | trajectories: {self.trajectories_num}'
        totals_string += f' | iterations: {self.iterations_num}'
        self.totals_string = totals_string

        # signaling totals updated
        self.signal_totals_updated()

######################################################################
# defining auxiliary functions


def generate_segmentation_masks(input_folder: str,
                                images_extension: str,
                                output_folder: str,
                                progress_tracker: ModuleProgressTracker
                                ) -> None:
    """
    Given a path to a folder containing
    binary masks, generates segmentation
    masks, saving output to given folder.
    """
    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=input_folder,
                                          extension=images_extension)

    # iterating over images in input folder
    for image_name in images:

        # updating progress tracker attributes
        progress_tracker.current_iteration += 1

        if progress_tracker.current_iteration < 24:
            continue

        # getting current image input/output paths
        current_input_path = join(input_folder,
                                  image_name)
        current_output_path = join(output_folder,
                                   image_name)

        # generating segmentation mask for current image
        generate_segmentation_mask(input_path=current_input_path,
                                   output_path=current_output_path)


def parse_and_run(args_dict: dict,
                  progress_tracker: ModuleProgressTracker
                  ) -> None:
    """
    Extracts args from args_dict
    and runs module function.
    """
    # getting input folder
    input_folder = args_dict['input_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output folder
    output_folder = args_dict['output_folder']

    # running generate_segmentation_masks function
    generate_segmentation_masks(input_folder=input_folder,
                                images_extension=images_extension,
                                output_folder=output_folder,
                                progress_tracker=progress_tracker)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # initializing current module progress tracker instance
    progress_tracker = ModuleProgressTracker()

    # running code in separate thread
    progress_tracker.run(function=parse_and_run,
                         args_parser=get_args_dict)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
