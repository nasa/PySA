# Author: Salvatore Mandra (salvatore.mandra@nasa.gov)
#
# Copyright © 2023, United States Government, as represented by the Administrator
# of the National Aeronautics and Space Administration. All rights reserved.
#
# The PySA, a powerful tool for solving optimization problems is licensed under
# the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
from typer import Typer, Argument, Option, BadParameter, Context
from typing_extensions import Annotated, Optional
from ._globals import __params
from ._sat import sat

__all__ = ['app']

# Description of PySA-DPLL
__description__ = "[bold][green]PySA-DPLL[/green][/bold]"

# Initialize this app
app = Typer(no_args_is_help=True, help=__description__, rich_markup_mode="rich")
app.command(help="[bold]Find all configurations with a "
            "[green]maximum number of unsatisfied clauses[/green][/bold]")(sat)


# Common command
@app.callback()
def main(
    filename: Annotated[str,
                        Option("--filename",
                               "-f",
                               help="Filename to use. Otherwise use stdin.",
                               show_default=False)] = None,
    walltime: Annotated[
        float, Option("--walltime", "-w", help="Walltime in seconds.")] = None,
    n_threads: Annotated[
        int,
        Option(
            "--n-threads",
            help=
            "Number of threads to use (by default, all available cores are used)."
        )] = None,
    stop_on_first: Annotated[
        bool,
        Option("--stop-on-first",
               help="Stop search when the first solution is found.")] = False,
    verbose: Annotated[
        bool, Option("--verbose", "-v", help="Verbose output.")] = False):
    # Update parameters
    __params.update(locals())
