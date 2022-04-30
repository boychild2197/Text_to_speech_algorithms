/*
This file is part of Talkie -- text-to-speech browser extension button.
<https://joelpurra.com/projects/talkie/>

Copyright (c) 2016, 2017, 2018, 2019, 2020, 2021 Joel Purra <https://joelpurra.com/>

Talkie is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Talkie is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Talkie.  If not, see <https://www.gnu.org/licenses/>.
*/

import React from "react";
import PropTypes from "prop-types";

import Icon from "./icon.jsx";

export default class TalkieEditionIcon extends React.PureComponent {
    static defaultProps = {
        mode: "inline",
        size: "1.3em",
        marginLeft: "0.3em",
        marginRight: "0.3em",
        className: "",
        network: null,
    };

    static propTypes = {
        mode: PropTypes.oneOf(["inline", "standalone"]).isRequired,
        size: PropTypes.string.isRequired,
        marginLeft: PropTypes.string.isRequired,
        marginRight: PropTypes.string.isRequired,
        className: PropTypes.string.isRequired,
        network: PropTypes.oneOf(["twitter", "facebook", "googleplus", "linkedin"]).isRequired,
    }

    render() {
        const {
            mode,
            size,
            marginLeft,
            marginRight,
            className,
            network,
        } = this.props;

        const networkClassName = `icon-${network}`;

        const classNames = [
            "icon-share",
            networkClassName,
            className,
        ]
            .join(" ")
            .trim();

        return (
            <Icon
                mode={mode}
                size={size}
                marginLeft={marginLeft}
                marginRight={marginRight}
                className={classNames}
            />
        );
    }
}
