# Copyright 2020 Fabio Tosi, Filippo Aleotti, Pierluigi Zama Ramirez, Matteo Poggi,
# Samuele Salti, Luigi Di Stefano, Stefano Mattoccia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from testers import general_tester


class Tester(general_tester.GeneralTester):
    """Error tester. If selected, it means that
        no valid Tester exist for that dataset/task
        association.
    """

    def test(self, network, dataloader):
        """This component has to raise ValueError, because
            that dataset/task association is not admitted.
            :param network: built Network
            :param dataloader: built Dataloader
            :raise ValueError: No testing for task are available for the selected dataset
        """
        raise ValueError("No testing for task are available for the selected dataset")
